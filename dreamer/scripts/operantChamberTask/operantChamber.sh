xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTask.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-train \
  --dreamer-args "--run.steps 2000000"
