xvfb-run -a python train.py \
  --task ../configs/foragingTask/foragingTaskSpawnerTree.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/foraging/foraging-train \
  --dreamer-args "--run.steps 1000000"
