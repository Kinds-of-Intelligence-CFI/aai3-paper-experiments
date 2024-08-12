xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTask.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamberCurriculum-eval \
  --from-checkpoint ./logdir/operantChamber/operantChamber-E-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 500000"
# 500k steps is at least 100 eval episodes at max 3k steps per episode
