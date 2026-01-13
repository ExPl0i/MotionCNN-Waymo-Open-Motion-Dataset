docker run -d --name motioncnn \
  --gpus all \
  --ipc=host --shm-size=8g \
  -v /mnt/scsi_nas/waymo_datasets:/data/waymo \
  -v /mnt/scsi_nas/env-predict/preprocessed:/data/preprocessed \
  -v /mnt/scsi_nas/env-predict/checkpoints:/data/checkpoints \
  motioncnn:cu117 \
  tail -f /dev/null
