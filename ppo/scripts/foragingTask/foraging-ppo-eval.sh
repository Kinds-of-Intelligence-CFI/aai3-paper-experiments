python evaluate_from_sh.py \
  --aai-env-path /Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/aai/env/AnimalAI.app \
  --model-save-path logdir_important_checkpoints/foraging/foraging-train/ppo/training-1000000.0 \
  --arenas-dir-path ../configs/foragingTask \
  --eval-csv-results-path results/foraging-eval.csv \
  --timescale 1 \
  --num-evals-per-instance 100 \
# Note: env and logdir should be provided as absolute paths if on MacOS

# TODO: turn the arenas-dir-path into a directory path, not a yaml OR support YAMLs as entries in evaluate script.