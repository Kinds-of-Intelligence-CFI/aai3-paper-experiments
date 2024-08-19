xvfb-run -a python train_from_sh.py \
  --task ../configs/whatWhereWhenTask/whatWhereWhenTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/whatWhereWhen/whatWhereWhen-train \
  --timesteps 4000000 \
  --aai_timescale 1 \
  --algorithm ppo
