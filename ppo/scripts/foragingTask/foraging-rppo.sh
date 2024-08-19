python -m train_from_sh \
  --task ../configs/foragingTask/foragingTaskSpawnerTree.yml \
  --env /Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/aai/env/AnimalAI.app \
  --logdir /Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/logdir/foraging/foraging-train/rppo \
  --timesteps 1000000 \
  --aai_timescale 1 \
  --algorithm recurrent_ppo \

# Note: env and logdir should be provided as absolute paths if on MacOS