xvfb-run -a python evaluate_from_sh.py \
  --aai-env-path ./aai/env/env3.1.3/AAI.x86_64 \
  --model-save-path logdir/operantChamber/operantChamber-train/training-2000000.0 \
  --arenas-dir-path aai/configs/operantChamber \
  --eval-csv-results-path results/operantChamber/ppo/all.csv \
  --timescale 1 \
  --num-evals-per-instance 100 \
  
# Copied arenas_dir_path to recurrent-ppo folder because current evaluate function implementation requires the target YAML file(s) to be the only file(s) in the configuration folder being passed.
