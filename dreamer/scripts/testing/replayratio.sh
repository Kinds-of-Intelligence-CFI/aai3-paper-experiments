#!/bin/bash

set -e
set -x
set -o pipefail
set -u

RATIO=$1

xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-curriculum.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/ratio/rr$RATIO \
  --dreamer-args "--run.steps 8e6 --run.train_ratio $RATIO"