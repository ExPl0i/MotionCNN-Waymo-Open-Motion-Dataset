# metrics_ap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style AP: 101-point interpolated precision."""
    if precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # precision envelope
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
    n_gt: int,
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


def compute_map_metrics(
    conf_per_mode: torch.Tensor,   # [B, K] logits or probs
    fde_per_mode: torch.Tensor,    # [B, K] meters
    gt_valid: torch.Tensor,        # [B] bool
    sample_ids: torch.Tensor,      # [B] int64 unique per GT
    conf_threshold: float = 0.5,
    dist_thresholds_m: Iterable[float] = tuple(np.arange(0.50, 0.96, 0.05)),
) -> Dict[str, float]:
    """
    Returns:
      precision, recall computed for top-1 mode at dist_thr=0.50m and conf_thr=conf_threshold
      mAP50 and mAP50-95 computed across ALL modes using confidences for ranking.
    """
    device = conf_per_mode.device
    gt_valid = gt_valid.bool()

    # If logits -> to probs for ranking/thresholding
    if conf_per_mode.min() < 0 or conf_per_mode.max() > 1:
        conf_probs = torch.softmax(conf_per_mode, dim=-1)
    else:
        conf_probs = conf_per_mode

    # ----- Precision / Recall (top-1) -----
    top_conf, top_idx = conf_probs.max(dim=-1)  # [B]
    top_fde = fde_per_mode.gather(1, top_idx.unsqueeze(1)).squeeze(1)  # [B]

    valid_mask = gt_valid & torch.isfinite(top_fde) & torch.isfinite(top_conf)
    n_gt = int(valid_mask.sum().item())

    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}

    dist_thr_50 = 0.50
    pred_pos = valid_mask & (top_conf >= conf_threshold)
    tp = int((pred_pos & (top_fde <= dist_thr_50)).sum().item())
    fp = int((pred_pos & (top_fde > dist_thr_50)).sum().item())
    fn = n_gt - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    # ----- mAP (all modes) -----
    # Flatten detections: each mode is a "detection" for that GT
    # conf_flat: [B*K], fde_flat: [B*K], sid_flat: [B*K]
    B, K = conf_probs.shape
    sid = sample_ids.to(device=device, dtype=torch.int64)
    sid_flat = sid.unsqueeze(1).repeat(1, K).reshape(-1)
    conf_flat = conf_probs.reshape(-1)
    fde_flat = fde_per_mode.reshape(-1)

    # filter invalid GTs entirely
    gt_valid_flat = gt_valid.unsqueeze(1).repeat(1, K).reshape(-1)
    sid_flat = sid_flat[gt_valid_flat]
    conf_flat = conf_flat[gt_valid_flat]
    fde_flat = fde_flat[gt_valid_flat]

    conf_np = conf_flat.detach().float().cpu().numpy()
    fde_np = fde_flat.detach().float().cpu().numpy()
    sid_np = sid_flat.detach().cpu().numpy()
    n_gt = int(gt_valid.sum().item())

    thresholds = list(dist_thresholds_m)
    aps = []
    for t in thresholds:
        aps.append(_compute_ap_for_threshold(conf_np, fde_np, sid_np, float(t), n_gt))

    mAP50 = aps[0]  # first threshold is 0.50 by умолчанию
    mAP50_95 = float(np.mean(aps)) if aps else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP50_95),
    }
