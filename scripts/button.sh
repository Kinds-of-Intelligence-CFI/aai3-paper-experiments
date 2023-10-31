xvfb-run -a python train.py \
  --task ./aai/configs/paper/buttonPressGreenCurriculum.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/button-train \
  --dreamer-args "--run.steps 2000000"
