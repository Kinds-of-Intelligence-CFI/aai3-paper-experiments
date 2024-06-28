DATETIME=$(date '+%Y%m%d-%H%M%S')
xvfb-run -a python train.py \
  --task ./aai/configs/paper/foragingTaskSpawnerTree.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/test-$DATETIME \
  --dreamer-args "--run.steps 2000"
