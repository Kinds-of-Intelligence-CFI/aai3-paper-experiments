# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/aai-competition-curriculum.yml \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-train-long12m \
#   --from-checkpoint ./logdir/competition-train-long/checkpoint.ckpt \
#   --dreamer-args "--run.steps 12e6" 

# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/competition1 \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-long12m-eval-ind1 \
#   --from-checkpoint ./logdir/competition-train-long12m/checkpoint.ckpt \
#   --eval-mode \
#   --dreamer-args "--run.steps 0.3e6"

# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/competition2 \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-long12m-eval-ind2 \
#   --from-checkpoint ./logdir/competition-train-long12m/checkpoint.ckpt \
#   --eval-mode \
#   --dreamer-args "--run.steps 0.3e6"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition3 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-long12m/competition-long12m-eval-ind3 \
  --from-checkpoint ./logdir/competition-long12m/competition-train-long12m/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6"
