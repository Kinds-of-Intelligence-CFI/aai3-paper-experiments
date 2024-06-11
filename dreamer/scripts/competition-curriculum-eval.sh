CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition1 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-eval-ind1 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 900000"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition2 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-eval-ind2 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 900000"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition3 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-eval-ind3 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 900000"