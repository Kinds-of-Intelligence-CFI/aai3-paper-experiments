from src.animalai_stable_baselines.train import train

from typing import *
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
import logging


@dataclass
class Args:
    task: Path
    env: Path
    timesteps: int
    from_checkpoint: Optional[Path]
    logdir: Optional[Path]
    aai_timescale: Optional[int]
    algorithm: Optional[str]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=Path, required=True, help='Path to the task file.')
    parser.add_argument('--env', type=Path, required=True, help='Path to the AnimalAI executable.')
    parser.add_argument('--from-checkpoint', type=Path, help='Load a checkpoint to continue training or evaluate from.')
    parser.add_argument('--logdir', type=Path, help='Directory to save logs to.')
    parser.add_argument('--timesteps', type=int, required=True, help='Number of steps to train agent on.')
    parser.add_argument('--aai_timescale', type=int, required=False, default=1,
                        help='The timescale to run AAI at. Defaults to 1, the human-play timescale.')
    parser.add_argument("--algorithm", type=str, required=False, default="ppo")
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    print(args)

    try:

        train(task=args.task,
              env=args.env,
              from_checkpoint=args.from_checkpoint,
              logdir=args.logdir,
              timesteps=args.timesteps,
              aai_timescale=args.aai_timescale,
              algorithm=args.algorithm,
              observations="camera",
              resolution=64,
              numsaves=20,
              wandb=True,
              inference=False)
    except Exception as e:
        logging.error(f"Exception: {e}")
        sys.exit(1)
        raise e


if __name__ == '__main__':
    main()
