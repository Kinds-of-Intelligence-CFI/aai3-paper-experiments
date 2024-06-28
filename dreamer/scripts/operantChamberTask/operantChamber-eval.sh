xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-eval \
  --from-checkpoint ./logdir/operantChamber/operantChamber-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 500000"
# 500k steps is at least 100 eval episodes at max 3k steps per episode
