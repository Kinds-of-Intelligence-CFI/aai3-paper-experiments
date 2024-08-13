xvfb-run -a python evaluate_from_sh.py \
  --aai-env-path ./aai/env/env3.1.3/AAI.x86_64 \
  --model-save-path logdir/ppo/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/training-5000000.0 \
  --arenas-dir-path ../configs/competition \
  --eval-csv-results-path results/competition/ppo/all.csv \
  --timescale 300 \
  --num-evals-per-instance 1 \
