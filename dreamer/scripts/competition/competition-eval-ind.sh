# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/competition1 \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-eval-ind1 \
#   --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
#   --eval-mode \
#   --dreamer-args "--run.steps 300000"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition2 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-eval-ind2 \
  --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 300000"

# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/competition3 \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-eval-ind3 \
#   --from-checkpoint ./logdir/competition-train/checkpoint.ckpt \
#   --eval-mode \
#   --dreamer-args "--run.steps 300000"