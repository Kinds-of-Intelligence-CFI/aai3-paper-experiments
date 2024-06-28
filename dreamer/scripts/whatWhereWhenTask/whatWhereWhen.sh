xvfb-run -a python train.py \
  --task ../configs/whatWhereWhenTask/whatWhereWhenTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/whatWhereWhen/whatWhereWhen-train \
  --dreamer-args "--run.steps 4000000"