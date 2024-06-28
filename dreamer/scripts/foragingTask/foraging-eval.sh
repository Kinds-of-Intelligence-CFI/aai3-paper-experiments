xvfb-run -a python train.py \
  --task ../configs/foragingTask/foragingTaskSpawnerTree.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/foraging/foraging-eval \
  --from-checkpoint ./logdir/foraging-train/checkpoint.ckpt \
  --eval-mode \
  --dreamer-args "--run.steps 500000"
# 500k steps is at least 100 eval episodes at max 3k steps per episode
