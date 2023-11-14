xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-curriculum.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-train \
  --dreamer-args "--run.steps 5000000"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-curriculum.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-eval \
  --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 15000000"
# 500k steps is at least 100 eval episodes at max 3k steps per episode
# There is about 300 arena's in the competition environment, which would make 300*500k = 150M steps.
# So lets do 10 episodes at max 3k, which would be 15M steps, more reasonable.
