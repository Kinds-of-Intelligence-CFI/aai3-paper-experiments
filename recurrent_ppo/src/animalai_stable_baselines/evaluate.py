from __future__ import annotations

import csv
import os
import random
import time
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from numpy.typing import NDArray

from animalai.environment import AnimalAIEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from animalai_stable_baselines.yaml_handling import find_yaml_files, yaml_combinor


def evaluate(aai_env_path: str,
             model_save_path: str,
             arenas_dir_path: str,
             eval_csv_results_path: str,
             load: BaseAlgorithm.load,
             use_camera: bool,
             resolution: int,
             use_ray_casts: bool,
             timescale: int,
             temp_batch_arenas_dir_path: str = "./",
             batch_size: int = 1,
             agent_inference: bool = False,
             base_port: int = 6600,
             verbose: bool = True,
             random_seed: int = 0,
             first_arena_ix_to_test: int = 0,
             save_step_results: bool = False,
             no_graphics: bool = False,
             deterministic_prediction: bool = True,
             num_evals_per_instance: int = 1,
             ) -> None:
    assert os.path.exists(aai_env_path)
    assert os.path.exists(model_save_path)
    assert os.path.exists(arenas_dir_path)

    arena_file_paths, arena_names = find_yaml_files(arenas_dir_path)
    model = load(model_save_path)
    random.seed(random_seed)
    last_arena_ix_to_test = len(arena_file_paths)

    for batch_start_ix in range(first_arena_ix_to_test, last_arena_ix_to_test, batch_size):
        port = _increment_port_number(base_port, 1)

        batch_counter = (batch_start_ix - first_arena_ix_to_test) // batch_size
        batch_end_ix = batch_start_ix + batch_size

        if batch_end_ix > last_arena_ix_to_test:
            batch_end_ix = last_arena_ix_to_test
        else:
            batch_end_ix = batch_end_ix

        if verbose:
            print(f"Running inferences on batch {batch_counter + 1}, composed of {batch_size} file(s). "
                  f"There is a total of {len(arena_file_paths)} files to test")

        if batch_size == 1:
            config_file_path = arena_file_paths[batch_start_ix]
        else:
            batch_files = arena_file_paths[batch_start_ix:batch_end_ix]
            temp_batch_file_name = f"TempConfig_{random_seed}_{batch_start_ix}.yml"
            config_file_path = yaml_combinor(file_list=batch_files,
                                             temp_file_location=temp_batch_arenas_dir_path,
                                             stored_file_name=temp_batch_file_name)

        if verbose:
            print("Opening AAI Environment.")

        env = AnimalAIEnvironment(
            inference=agent_inference,  # Set true when watching the agent
            seed=random_seed,
            worker_id=random_seed,
            file_name=aai_env_path,
            arenas_configurations=config_file_path,
            base_port=port,
            useCamera=use_camera,
            resolution=resolution,
            useRayCasts=use_ray_casts,
            timescale=timescale,
            no_graphics=no_graphics
        )
        env = UnityToGymWrapper(env,
                                uint8_visual=True,  # TODO: make a condition on use_camera
                                allow_multiple_obs=False,
                                flatten_branched=True)

        model.set_env(env)

        obs = env.reset()

        for instance_ix in range(batch_start_ix, batch_end_ix):
            for _ in range(num_evals_per_instance):
                arena_name = arena_names[instance_ix]
                done = False
                episode_reward = 0
                step_counter = 0

                while not done:
                    action, _state = model.predict(obs, deterministic=deterministic_prediction, )
                    # print(action)
                    obs, reward, done, info = env.step(action.item())
                    episode_reward += reward
                    step_counter += 1
                    env.render()

                    if save_step_results:
                        if verbose:
                            print(f"Writing observation data for step {step_counter} of the '{arena_name}' arena.")
                        obs_labels = ["x_velocity",
                                      "y_velocity",
                                      "z_velocity",
                                      "x_pos",
                                      "y_pos",
                                      "z_pos"]  # Extend this variable as needed
                        non_obs_labels = ["arena_name", "step_counter", "step_reward", "episode_reward"]
                        col_data = [arena_name, step_counter, reward, episode_reward] + _get_obs_data_from_labels(
                            obs_labels, obs)
                        col_labels = non_obs_labels + obs_labels
                        _update_results_csv(results_csv_path=eval_csv_results_path,
                                            column_labels=col_labels,
                                            column_data=col_data)

                    if done:
                        if verbose:
                            print(f"Episode Reward: {episode_reward}")
                        obs = env.reset()
                        col_labels = ["arena_name", "episode_reward"]
                        col_data = [arena_name, episode_reward]
                        _update_results_csv(results_csv_path=eval_csv_results_path,
                                            column_labels=col_labels,
                                            column_data=col_data)
                        break
        env.close()
        if verbose:
            print("Moving to next batch.")


POSSIBLE_OBS_KEYS = ["x_velocity", "y_velocity", "z_velocity", "x_pos", "y_pos", "z_pos"]


def _get_obs_data_from_labels(column_labels: List[str],
                              observation: Union[List[NDArray], NDArray]) -> List[str]:
    assert all(element in POSSIBLE_OBS_KEYS for element in column_labels)
    column_labels_to_data = {
        "x_velocity": observation[1][1],
        "y_velocity": observation[1][2],
        "z_velocity": observation[1][3],
        "x_pos": observation[1][4],
        "y_pos": observation[1][5],
        "z_pos": observation[1][6],
    }
    return [column_labels_to_data[label] for label in column_labels]


def _update_results_csv(results_csv_path: str,
                        column_labels: List[str],
                        column_data: List[str]) -> None:
    file_exists = os.path.isfile(results_csv_path)
    with open(results_csv_path, 'a' if file_exists else 'w', newline='') as csv_file:
        csv_write = csv.writer(csv_file)
        if not file_exists:
            csv_write.writerow(column_labels)
        csv_write.writerow(column_data)


def _increment_port_number(base_port: int,
                           increment: int) -> int:
    return base_port + increment


def example():
    evaluate(aai_env_path="/Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/aai/env"
                          "/AnimalAI.app",
             model_save_path="logdir_important_checkpoints/ppo-niall_L1_with_sanity_green-2M_steps/training"
                             "-2024_07_17_23_19/training-1000000.0",
             arenas_dir_path="aai/configs/ignore-sanity_green",
             eval_csv_results_path=f"results/ignore/ignore.csv",
             load=RecurrentPPO.load,
             use_camera=True,
             resolution=64,
             use_ray_casts=False,
             timescale=1,
             agent_inference=True,
             save_step_results=False,
             deterministic_prediction=False,
             num_evals_per_instance=3,
             )


if __name__ == "__main__":
    example()
