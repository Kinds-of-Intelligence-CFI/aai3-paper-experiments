# Train dreamer on curriculum, working through levels of AAI testbed.

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1 \
  --dreamer-args "--run.steps 5000000"


CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_2.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_2 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_3.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_3 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_2/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_4.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_4 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_3/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"


CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_5.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_5 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_4/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_6.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_6 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_5/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_7.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_7 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_6/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_8.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_8 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_7/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_9.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_9 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_8/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_10.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_10 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_9/checkpoint.ckpt \
  --dreamer-args "--run.steps 5000000"
