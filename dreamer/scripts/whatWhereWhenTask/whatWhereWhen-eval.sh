xvfb-run -a python train.py \
  --task ../configs/whatWhereWhenTask/whatWhereWhenTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/whatWhereWhen/whatWhereWhen-eval \
  --from-checkpoint ./logdir/whatWhereWhen/whatWhereWhen-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 600000"
# 600k steps is at least 100 eval episodes at max 4k steps per episode
