xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-A.yml \
  --env ../env/AAI.x86_64 \
  --logdir ./logdir/operantChamber/operantChamber-A-train \
  --dreamer-args "--run.steps 400000"

xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-B.yml \
  --env ../env/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-A-train/checkpoint.ckpt \
  --logdir ./logdir/operantChamber/operantChamber-B-train \
  --dreamer-args "--run.steps 400000"

xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-C.yml \
  --env ../env/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-B-train/checkpoint.ckpt \
  --logdir ./logdir/operantChamber/operantChamber-C-train \
  --dreamer-args "--run.steps 400000"

xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-D.yml \
  --env ../env/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-C-train/checkpoint.ckpt \
  --logdir ./logdir/operantChamber/operantChamber-D-train \
  --dreamer-args "--run.steps 400000"

xvfb-run -a python train.py \
  --task ../configs/operantChamberTask/operantChamberTaskCurriculum-E.yml \
  --env ../env/AAI.x86_64 \
  --from-checkpoint ./logdir/operantChamber/operantChamber-D-train/checkpoint.ckpt \
  --logdir ./logdir/operantChamber/operantChamber-E-train \
  --dreamer-args "--run.steps 400000"