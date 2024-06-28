xvfb-run -a python train.py \
  --task ../configs/foragingTask/foragingTaskSpawnerTree.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/foraging/foraging-train \
  --dreamer-args "--run.steps 1000000"
