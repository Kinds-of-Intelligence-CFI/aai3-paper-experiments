import sys
import argparse
from dataclasses import dataclass
import logging

from stable_baselines3 import PPO

from src.animalai_stable_baselines.evaluate import evaluate


@dataclass
class Args:
    aai_env_path: str
    model_save_path: str
    arenas_dir_path: str
    eval_csv_results_path: str
    timescale: int
    num_evals_per_instance: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aai-env-path', type=str, required=True, help='Path to the AnimalAI executable.')
    parser.add_argument('--model-save-path', type=str, required=True, help="Path to saved model to be evaluated.")
    parser.add_argument('--arenas-dir-path', type=str, required=True,
                        help='Path to directory containing YAML configs to test the saved model on')
    parser.add_argument('--eval-csv-results-path', type=str, required=True,
                        help="Path to results CSV including .csv extension.")
    parser.add_argument('--timescale', type=int, required=True, help='The timescale to run AAI at.')
    parser.add_argument('--num-evals-per-instance', type=int, required=False,
                        help='Number of evaluations per arena. Defaults to 1.')
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    print(args)

    try:
        evaluate(aai_env_path=args.aai_env_path,
                 model_save_path=args.model_save_path,
                 arenas_dir_path=args.arenas_dir_path,
                 eval_csv_results_path=args.eval_csv_results_path,
                 timescale=args.timescale,
                 num_evals_per_instance=args.num_evals_per_instance,
                 load=PPO.load,
                 use_camera=True,
                 resolution=64,
                 use_ray_casts=False,
                 agent_inference=False,
                 save_step_results=False,
                 deterministic_prediction=False,
                 batch_size=100)
    except Exception as e:
        logging.error(f"Exception: {e}")
        sys.exit(1)
        raise e


if __name__ == '__main__':
    main()
