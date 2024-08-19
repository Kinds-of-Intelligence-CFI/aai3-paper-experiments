xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTask.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-train \
  --timesteps 2000000 \
  --aai_timescale 1 \
  --algorithm ppo \
