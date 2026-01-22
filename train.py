# train.py
import os
import argparse
from glob import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm

from utils import get_config
from losses import NLLGaussian2d


# ----------------------------
# Dataset
# ----------------------------
class MotionCNNDataset(Dataset):
    def __init__(self, data_path, load_roadgraph: bool = False) -> None:
        super().__init__()
        self._load_roadgraph = load_roadgraph
        self._files = glob(os.path.join(data_path, '*', 'agent_data', '*.npz'))
        self._roadgraph_data = glob(os.path.join(data_path, '*', 'roadgraph_data', 'segments_global.npz'))
        self._scid_to_roadgraph = {f.split('/')[-3]: f for f in self._roadgraph_data}

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        data = dict(np.load(self._files[idx], allow_pickle=True))
        if self._load_roadgraph:
            roadgraph_data_file = self._scid_to_roadgraph[data['scenario_id'].item()]
            roadgraph_data = np.load(roadgraph_data_file)['roadgraph_segments']
            roadgraph_valid = np.ones(roadgraph_data.shape[0])
            n_to_pad = 6000 - roadgraph_data.shape[0]
            roadgraph_data = np.pad(roadgraph_data, ((0, n_to_pad), (0, 0), (0, 0)))
            roadgraph_valid = np.pad(roadgraph_valid, (0, n_to_pad))
            data['roadgraph_data'] = roadgraph_data
            data['roadgraph_valid'] = roadgraph_valid

        data['raster'] = data['raster'].transpose(2, 0, 1) / 255.0
        data['scenario_id'] = data['scenario_id'].item()
        return data


def dict_to_cuda(data_dict: Dict[str, torch.Tensor], device: torch.device):
    gpu_required_keys = ['raster', 'future_valid', 'future_local']
    for key in gpu_required_keys:
        data_dict[key] = data_dict[key].to(device, non_blocking=True)
    return data_dict


# ----------------------------
# Model
# ----------------------------
def get_model(model_config):
    # x, y, sigma_xx, sigma_yy, visibility
    n_components = 5
    n_modes = model_config['n_modes']
    n_timestamps = model_config['n_timestamps']
    output_dim = n_modes + n_modes * n_timestamps * n_components
    model = timm.create_model(
        model_config['backbone'],
        pretrained=True,
        in_chans=27,
        num_classes=output_dim
    )
    return model


def limited_softplus(x):
    return torch.clamp(F.softplus(x), min=0.1, max=10)


def postprocess_predictions(predicted_tensor, model_config):
    confidences = predicted_tensor[:, :model_config['n_modes']]
    components = predicted_tensor[:, model_config['n_modes']:]
    components = components.reshape(-1, model_config['n_modes'], model_config['n_timestamps'], 5)
    sigma_xx = components[:, :, :, 2:3]
    sigma_yy = components[:, :, :, 3:4]
    visibility = components[:, :, :, 4:]
    return {
        'confidences': confidences,  # logits
        'xy': components[:, :, :, :2],
        'sigma_xx': limited_softplus(sigma_xx) if model_config['predict_covariances'] else torch.ones_like(sigma_xx),
        'sigma_yy': limited_softplus(sigma_yy) if model_config['predict_covariances'] else torch.ones_like(sigma_yy),
        'visibility': visibility
    }


# ----------------------------
# DDP helpers
# ----------------------------
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_setup():
    if not is_dist():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())


def ddp_cleanup():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


# ----------------------------
# Metrics: Precision / Recall / mAP50 / mAP50-95 (по порогам FDE в метрах)
# ----------------------------
def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style AP: 101-point interpolated precision."""
    if precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in recall_levels:
        p = mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0
        ap += p / 101.0
    return float(ap)


def _compute_ap_for_threshold(
    conf: np.ndarray,
    fde: np.ndarray,
    sample_ids: np.ndarray,
    dist_thr_m: float,
    n_gt: int
) -> float:
    """Each GT is one sample_id. TP if (fde<=thr) and GT not matched yet."""
    order = np.argsort(-conf)
    matched = set()

    tps = np.zeros(order.shape[0], dtype=np.float32)
    fps = np.zeros(order.shape[0], dtype=np.float32)

    for j, idx in enumerate(order):
        sid = int(sample_ids[idx])
        if sid in matched:
            fps[j] = 1.0
            continue
        if fde[idx] <= dist_thr_m:
            tps[j] = 1.0
            matched.add(sid)
        else:
            fps[j] = 1.0

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    recall = tp_cum / max(1, n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    return _ap_from_pr(precision, recall)


@torch.no_grad()
def compute_map_metrics(
    conf_per_mode: torch.Tensor,   # [N, K] logits or probs
    fde_per_mode: torch.Tensor,    # [N, K] meters
    gt_valid: torch.Tensor,        # [N] bool
    sample_ids: torch.Tensor,      # [N] int64 unique per GT
    conf_threshold: float = 0.5,
    dist_thresholds_m: Iterable[float] = tuple(np.arange(0.50, 0.96, 0.05)),
) -> Dict[str, float]:
    # logits -> probs
    if conf_per_mode.min() < 0 or conf_per_mode.max() > 1:
        conf_probs = torch.softmax(conf_per_mode, dim=-1)
    else:
        conf_probs = conf_per_mode

    gt_valid = gt_valid.bool()
    valid_mask = gt_valid & torch.isfinite(conf_probs).all(dim=1) & torch.isfinite(fde_per_mode).all(dim=1)
    n_gt = int(valid_mask.sum().item())
    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}

    # Precision/Recall: top-1 mode, threshold by conf_threshold, TP if FDE<=0.50m
    top_conf, top_idx = conf_probs.max(dim=-1)  # [N]
    top_fde = fde_per_mode.gather(1, top_idx.unsqueeze(1)).squeeze(1)  # [N]
    dist_thr_50 = 0.50

    pred_pos = valid_mask & (top_conf >= conf_threshold)
    tp = int((pred_pos & (top_fde <= dist_thr_50)).sum().item())
    fp = int((pred_pos & (top_fde > dist_thr_50)).sum().item())
    fn = n_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    # mAP: each mode is a "detection"
    N, K = conf_probs.shape
    sid = sample_ids.to(dtype=torch.int64)
    sid_flat = sid.unsqueeze(1).repeat(1, K).reshape(-1)
    conf_flat = conf_probs.reshape(-1)
    fde_flat = fde_per_mode.reshape(-1)

    valid_flat = valid_mask.unsqueeze(1).repeat(1, K).reshape(-1)
    sid_flat = sid_flat[valid_flat]
    conf_flat = conf_flat[valid_flat]
    fde_flat = fde_flat[valid_flat]

    conf_np = conf_flat.detach().float().cpu().numpy()
    fde_np = fde_flat.detach().float().cpu().numpy()
    sid_np = sid_flat.detach().cpu().numpy()

    thresholds = list(dist_thresholds_m)
    aps = []
    for t in thresholds:
        aps.append(_compute_ap_for_threshold(conf_np, fde_np, sid_np, float(t), n_gt))

    mAP50 = aps[0] if len(aps) > 0 else 0.0
    mAP50_95 = float(np.mean(aps)) if len(aps) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP50_95),
    }


# ----------------------------
# CLI / checkpoint
# ----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data-path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--checkpoints-path", type=str, required=True, help="Path to checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--multi-gpu", action='store_true', help="Legacy DataParallel. For 2x4090 prefer torchrun+DDP.")
    args = parser.parse_args()
    return args


def get_last_checkpoint_file(path: str) -> Optional[str]:
    list_of_files = glob(f'{path}/*.pth')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # helps load DP/DDP checkpoints into plain model
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# ----------------------------
# Validation
# ----------------------------
@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_module: nn.Module,
    model_config: Dict,
    training_config: Dict,
    device: torch.device
) -> Dict[str, float]:
    model.eval()

    amp_enabled = bool(training_config.get("amp", True))
    amp_dtype = str(training_config.get("amp_dtype", "bf16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    total_loss_sum = 0.0
    total_count = 0

    all_conf_cpu: List[torch.Tensor] = []
    all_fde_cpu: List[torch.Tensor] = []
    all_valid_cpu: List[torch.Tensor] = []
    all_sid_cpu: List[torch.Tensor] = []

    sid_base = get_rank() * 1_000_000_000
    sid_cursor = 0

    val_pbar = tqdm(val_loader, desc="val", leave=False, disable=not is_main_process())
    for batch in val_pbar:
        batch = dict_to_cuda(batch, device)

        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=amp_enabled):
            pred_tensor = model(batch["raster"].float())
            pred_dict = postprocess_predictions(pred_tensor, model_config)
            loss = loss_module(batch, pred_dict)

        bs = batch["raster"].shape[0]
        total_loss_sum += float(loss.item()) * bs
        total_count += bs

        # FDE per mode (final timestep)
        pred_xy = pred_dict["xy"]              # [B, K, T, 2]
        gt_xy = batch["future_local"]          # [B, T, 2]
        valid = batch["future_valid"]          # [B, T] or [B, T, 1]

        if valid.ndim == 3:
            valid_last = valid[:, -1, 0].bool()
        else:
            valid_last = valid[:, -1].bool()

        pred_final = pred_xy[:, :, -1, :]              # [B, K, 2]
        gt_final = gt_xy[:, -1, :].unsqueeze(1)        # [B, 1, 2]
        fde = torch.linalg.vector_norm(pred_final - gt_final, dim=-1)  # [B, K]

        sid = torch.arange(bs, device=device, dtype=torch.int64) + (sid_base + sid_cursor)
        sid_cursor += bs

        all_conf_cpu.append(pred_dict["confidences"].detach().cpu())
        all_fde_cpu.append(fde.detach().cpu())
        all_valid_cpu.append(valid_last.detach().cpu())
        all_sid_cpu.append(sid.detach().cpu())

    conf = torch.cat(all_conf_cpu, dim=0)
    fde = torch.cat(all_fde_cpu, dim=0)
    gt_valid = torch.cat(all_valid_cpu, dim=0)
    sample_ids = torch.cat(all_sid_cpu, dim=0)

    # gather to rank0 for metrics
    if is_dist():
        obj = {
            "conf": conf,
            "fde": fde,
            "valid": gt_valid,
            "sid": sample_ids,
            "loss_sum": total_loss_sum,
            "count": total_count,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)

        if is_main_process():
            conf = torch.cat([g["conf"] for g in gathered], dim=0)
            fde = torch.cat([g["fde"] for g in gathered], dim=0)
            gt_valid = torch.cat([g["valid"] for g in gathered], dim=0)
            sample_ids = torch.cat([g["sid"] for g in gathered], dim=0)
            total_loss_sum = float(sum(g["loss_sum"] for g in gathered))
            total_count = int(sum(g["count"] for g in gathered))

        dist.barrier()

    val_loss = total_loss_sum / max(1, total_count)

    metrics_cfg = training_config.get("metrics", {})
    conf_thr = float(metrics_cfg.get("conf_threshold", 0.5))
    dist_thrs = metrics_cfg.get("dist_thresholds_m", list(np.arange(0.50, 0.96, 0.05)))

    if is_main_process():
        metrics = compute_map_metrics(
            conf_per_mode=conf,
            fde_per_mode=fde,
            gt_valid=gt_valid,
            sample_ids=sample_ids,
            conf_threshold=conf_thr,
            dist_thresholds_m=dist_thrs,
        )
        metrics["val_loss"] = float(val_loss)
        return metrics

    return {"val_loss": float(val_loss)}


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_arguments()
    ddp_setup()

    # Device
    local_rank = get_local_rank()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Speed knobs for RTX 4090 / Ada
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Config
    general_config = get_config(args.config)
    model_config = general_config['model']
    training_config = general_config['training']
    config_name = args.config.split('/')[-1].split('.')[0]

    # TensorBoard (rank0 only)
    log_dir = training_config.get("tensorboard_logdir", os.path.join(args.checkpoints_path, "runs", config_name))
    writer = SummaryWriter(log_dir=log_dir) if is_main_process() else None

    # Model / optimizer / loss
    model = get_model(model_config).to(device)
    optimizer = Adam(model.parameters(), **training_config['optimizer'])
    loss_module = NLLGaussian2d()

    # Resume
    processed_batches = 0
    epochs_processed = 0

    experiment_checkpoints_dir = os.path.join(args.checkpoints_path, config_name)
    if is_main_process() and (not os.path.exists(experiment_checkpoints_dir)):
        os.makedirs(experiment_checkpoints_dir, exist_ok=True)
    if is_dist():
        dist.barrier()

    latest_checkpoint = get_last_checkpoint_file(experiment_checkpoints_dir) if is_main_process() else None
    if is_dist():
        obj_list = [latest_checkpoint]
        dist.broadcast_object_list(obj_list, src=0)
        latest_checkpoint = obj_list[0]

    if latest_checkpoint is not None:
        if is_main_process():
            print(f"Loading checkpoint from {latest_checkpoint}")
        ckpt = torch.load(latest_checkpoint, map_location="cpu")
        state = _strip_module_prefix(ckpt.get("model_state_dict", {}))
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epochs_processed = int(ckpt.get("epochs_processed", 0))
        processed_batches = int(ckpt.get("processed_batches", 0))
        _move_optimizer_state_to_device(optimizer, device)

    # Wrap model
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Data
    train_dataset = MotionCNNDataset(args.train_data_path)
    val_dataset = MotionCNNDataset(args.val_data_path, load_roadgraph=True)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None and training_config["train_dataloader"].get("shuffle", True)),
        **training_config['train_dataloader']
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **training_config['val_dataloader']
    )

    # AMP
    amp_enabled = bool(training_config.get("amp", True))
    amp_dtype = str(training_config.get("amp_dtype", "bf16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_scaler = amp_enabled and (amp_dtype == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    accum = int(training_config.get("accumulate_grad_batches", 1))
    grad_clip = training_config.get("grad_clip_norm", None)
    log_every = int(training_config.get("log_every", 50))
    save_every_epochs = int(training_config.get("save_every_epochs", 1))

    global_step = processed_batches

    # Train loop
    for epoch in range(epochs_processed, int(training_config['num_epochs'])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, total=len(train_loader), disable=not is_main_process(), desc=f"epoch {epoch} train")
        for step, batch in enumerate(pbar):
            batch = dict_to_cuda(batch, device)

            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=amp_enabled):
                pred_tensor = model(batch['raster'].float())
                pred_dict = postprocess_predictions(pred_tensor, model_config)
                loss = loss_module(batch, pred_dict)
                loss = loss / accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0:
                if grad_clip is not None:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.item()) * accum
            running_losses.append(loss_value)
            global_step += 1
            processed_batches = global_step

            if is_main_process():
                pbar.set_description(f"epoch {epoch} train | loss {np.mean(running_losses[-100:]):.4f}")

                if writer is not None and (global_step % log_every == 0):
                    writer.add_scalar("train/loss", np.mean(running_losses[-100:]), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # --- Validation after each epoch ---
        metrics = validate_one_epoch(model, val_loader, loss_module, model_config, training_config, device)

        if is_main_process():
            print(
                f"[epoch {epoch}] "
                f"val_loss={metrics.get('val_loss', 0.0):.4f} "
                f"P={metrics.get('precision', 0.0):.4f} "
                f"R={metrics.get('recall', 0.0):.4f} "
                f"mAP50={metrics.get('mAP50', 0.0):.4f} "
                f"mAP50-95={metrics.get('mAP50_95', 0.0):.4f}"
            )
            if writer is not None and "precision" in metrics:
                writer.add_scalar("val/loss", metrics["val_loss"], epoch)
                writer.add_scalar("val/precision", metrics["precision"], epoch)
                writer.add_scalar("val/recall", metrics["recall"], epoch)
                writer.add_scalar("val/mAP50", metrics["mAP50"], epoch)
                writer.add_scalar("val/mAP50_95", metrics["mAP50_95"], epoch)

        # --- Save checkpoint ---
        if is_main_process() and (save_every_epochs > 0) and ((epoch + 1) % save_every_epochs == 0):
            # unwrap DP/DDP
            if isinstance(model, (nn.DataParallel, DDP)):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            ckpt = {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_processed": epoch + 1,
                "processed_batches": processed_batches,
                "metrics": metrics if is_main_process() else None,
            }
            last_path = os.path.join(experiment_checkpoints_dir, "last.pth")
            torch.save(ckpt, last_path)

            # optional epoch snapshot
            snap_path = os.path.join(experiment_checkpoints_dir, f"epoch_{epoch+1:03d}.pth")
            torch.save(ckpt, snap_path)

    if writer is not None:
        writer.close()
    ddp_cleanup()


if __name__ == '__main__':
    main()
