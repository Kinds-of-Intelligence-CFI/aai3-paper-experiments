xvfb-run -a python train.py \
  --task ../configs/whatWhereWhenTask/whatWhereWhenTask-Curriculum.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/whatWhereWhen/whatWhereWhen-curriculum-train \
  --dreamer-args "--run.steps 4000000"