# xvfb-run -a python train.py \
#   --task ./aai/configs/paper/aai-competition-curriculum.yml \
#   --env ./aai/env/env3.1.3/AAI.x86_64 \
#   --logdir ./logdir/competition-train-long \
#   --from-checkpoint ./logdir/competition/competition-train/checkpoint.ckpt \
#   --dreamer-args "--run.steps 80000000" 
# # woops (80m), should use scientific notation

xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition1 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-long8m-eval-ind1 \
  --from-checkpoint ./logdir/competition-train-long/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition2 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-long8m-eval-ind2 \
  --from-checkpoint ./logdir/competition-train-long/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6"

xvfb-run -a python train.py \
  --task ./aai/configs/paper/competition3 \
  --env ./aai/env/env3.1.3/AAI.x86_64 \
  --logdir ./logdir/competition-long8m-eval-ind3 \
  --from-checkpoint ./logdir/competition-train-long/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 0.3e6"
