CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition1 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-eval-ind1 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 300000" \
  --aai-timescale 300

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition2 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-eval-ind2 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 300000" \
  --aai-timescale 300

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition3 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-eval-ind3 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 300000" \
  --aai-timescale 300

#CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/competition-extra \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-eval-extra \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/checkpoint.ckpt \
#  --eval-mode \
#  --dreamer-args "--run.steps 64000" \
#  --aai-timescale 300
