xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-A-train \
  --timesteps 400000 \
  --aai_timescale 1 \
  --algorithm ppo

xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-A-train/training-400000.0 \
  --logdir ./logdir/operantChamber/operantChamber-B-train \
  --timesteps 400000 \
  --aai_timescale 1 \
  --algorithm ppo

xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-B-train/training-400000.0 \
  --logdir ./logdir/operantChamber/operantChamber-C-train \
  --timesteps 400000 \
  --aai_timescale 1 \
  --algorithm ppo

xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-C-train/training-400000.0 \
  --logdir ./logdir/operantChamber/operantChamber-D-train \
  --timesteps 400000 \
  --aai_timescale 1 \
  --algorithm ppo

xvfb-run -a python train_from_sh.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-D-train/training-400000.0 \
  --logdir ./logdir/operantChamber/operantChamber-E-train \
  --timesteps 400000 \
  --aai_timescale 1 \
  --algorithm ppo