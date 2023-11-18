xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-test-1.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-eval1 \
  --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 5000000"

