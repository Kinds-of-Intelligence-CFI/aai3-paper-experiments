xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-train \
  --dreamer-args "--run.steps 2000000"
