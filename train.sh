torchrun --standalone --nproc_per_node=2 train.py \
  --train-data-path /data/preprocessed/training \
  --val-data-path /data/preprocessed/validation \
  --checkpoints-path /data/checkpoints \
  --config configs/basic.yaml