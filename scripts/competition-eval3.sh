xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-test-3.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-eval3 \
  --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 5000000"