xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-test-1.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/ratio/16-eval-ind1 \
  --from-checkpoint ./logdir/ratio/16/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6 --run.train_ratio 16"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-test-2.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/ratio/16-eval-ind2 \
  --from-checkpoint ./logdir/ratio/16/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6 --run.train_ratio 16"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/aai-competition-test-3.yml \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/ratio/16-eval-ind3 \
  --from-checkpoint ./logdir/ratio/16/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6 --run.train_ratio 16"