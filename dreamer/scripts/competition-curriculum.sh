# Train dreamer on curriculum, working through levels of AAI testbed.

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1 \
#  --dreamer-args "--run.steps 2000000"

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_2.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_2 \
#  --dreamer-args "--run.steps 4000000" 

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_3.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_2/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_3 \
#  --dreamer-args "--run.steps 6000000"

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_4.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_3/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_4 \
#  --dreamer-args "--run.steps 8000000"

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_5.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_4/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_5 \
#  --dreamer-args "--run.steps 10000000"

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_6.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_5/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_6 \
#  --dreamer-args "--run.steps 12000000"

#CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#  --task ./aai/configs/paper/curriculumL1_7.yaml \
#  --env ./aai/env/env3.1.3/AAI.x86_64 \
#  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_6/checkpoint.ckpt \
#  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_7 \
#  --dreamer-args "--run.steps 14000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_8.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_7/checkpoint.ckpt \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_8 \
  --dreamer-args "--run.steps 16000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_9.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_8/checkpoint.ckpt \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_9 \
  --dreamer-args "--run.steps 18000000"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_10.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_9/checkpoint.ckpt \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_10 \
  --dreamer-args "--run.steps 20000000"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
  --task ./aai/configs/paper/curriculumL1_10.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-L1_L10/checkpoint.ckpt \
  --logdir ./logdir/competition-curriculum/competition-curriculum-L1_10_5M \
  --dreamer-args "--run.steps 25000000"
