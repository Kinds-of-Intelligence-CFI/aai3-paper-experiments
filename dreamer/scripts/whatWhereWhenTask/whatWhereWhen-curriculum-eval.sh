xvfb-run -a python train.py \
  --task ../configs/whatWhereWhenTask/whatWhereWhenTask-Curriculum.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/whatWhereWhen/whatWhereWhen-curriculum-eval \
  --from-checkpoint ./logdir/whatWhereWhen/whatWhereWhen-curriculum-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 600000"
# 600k steps is at least 100 eval episodes at max 4k steps per episode
