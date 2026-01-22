docker run -d --name motioncnn \
  --gpus '"device=0,2"' \
  --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=IPC_LOCK \
  -v /mnt/scsi_nas/waymo_datasets:/data/waymo \
  -v /mnt/scsi_nas/env-predict/preprocessed:/data/preprocessed \
  -v /mnt/scsi_nas/env-predict/checkpoints:/data/checkpoints \
  motioncnn:cu118 \
  tail -f /dev/null