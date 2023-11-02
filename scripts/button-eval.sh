xvfb-run -a python train.py \
  --task ./aai/configs/paper/buttonPressGreen.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/button-eval \
  --from-checkpoint ./logdir/button-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 500000"
# 500k steps is at least 100 eval episodes at max 3k steps per episode
