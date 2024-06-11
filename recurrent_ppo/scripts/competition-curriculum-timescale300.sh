  xvfb-run -a python train_from_sh.py \
  --task ./aai/configs/paper/curriculumL1.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1 \
  --timesteps 2000000 \
  --aai_timescale 300

   xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_2.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_2 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_3.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_2/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_3 \
   --number_steps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_4.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_3/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_4 \
   --timesteps 2000000  \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_5.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_4/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_5 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_6.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_5/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_6 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_7.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_6/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_7 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_8.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_7/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_8 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_9.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_8/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_9 \
   --timesteps 2000000 \
   --aai_timescale 300

  xvfb-run -a python train_from_sh.py \
   --task ./aai/configs/paper/curriculumL1_10.yaml \
   --env ./aai/env/env3.1.3/AAI.x86_64 \
   --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_9/checkpoint.ckpt \
   --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10 \
   --timesteps 2000000 \
   --aai_timescale 300

 xvfb-run -a python train_from_sh.py \
  --task ./aai/configs/paper/curriculumL1_10.yaml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --from-checkpoint ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10/checkpoint.ckpt \
  --logdir ./logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M \
  --timesteps 5000000 \
  --aai_timescale 300
