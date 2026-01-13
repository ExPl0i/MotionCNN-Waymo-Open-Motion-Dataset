python prerender.py \
    --data-path /data/waymo/motion_v_1_3_1/uncompressed/tf_example/training \
    --output-path /data/preprocessed/training \
    --config /workspace/configs/basic.yaml \
    --n-jobs 32 \
    --n-shards 8 \
    --shard-id 0 \