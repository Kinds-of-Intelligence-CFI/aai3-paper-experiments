# ruff: noqa: E402
import logging
import os
import random
from pathlib import Path

# Make sure this is above the import of AnimalAIEnvironment
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] [%(module)s] %(message)s",
    level=logging.INFO,
)

import stable_baselines3 as sb3
import torch as th
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from sb3_contrib import RecurrentPPO
from typing import Optional
import time

import animalai_stable_baselines.utils as utils


def algorithm_choice(algorithm_name):
    algorithm_name = algorithm_name.lower()
    algorithm_types = ['a2c', 'dqn', 'ppo', 'recurrent_ppo']
    if algorithm_name not in algorithm_types:
        raise ValueError(
            "Invalid algorithm type. The following have been explicitly tested in AAI: %s. Add further options from stable-baselines3 to the algorithm_choice function in the train.py file." % algorithm_types)
    else:
        algo_dict = {
            'a2c': sb3.A2C,
            'dqn': sb3.DQN,
            'ppo': sb3.PPO,
            'recurrent_ppo': RecurrentPPO
        }
    return algo_dict.get(algorithm_name.lower())


def train(task: Path,
          algorithm: str,
          observations: str,
          timesteps: int,
          resolution: int,
          env: Path = None,
          raycast_degrees: int = 60,
          numsaves: int = 1,
          from_checkpoint: Path = None,
          logdir: Path = None,
          aai_timescale: int = 300,
          grayscale: bool = False,
          no_graphics: bool = False,
          device: str = 'auto',
          wandb: bool = False,
          inference: bool = False):
    # Argument checks
    assert from_checkpoint.exists() if from_checkpoint is not None else True, f"Checkpoint not found: {from_checkpoint}."
    assert from_checkpoint.is_file() if from_checkpoint is not None else True, f"Checkpoint must be a file but is not: {from_checkpoint}."
    assert task.exists(), f"Task file not found: {task}."
    assert env.exists() if env is not None else True, f"AAI executable file not found: {env}."

    assert aai_timescale > 0, "The timescale cannot be 0. To ensure this works, it must be an integer"
    assert timesteps > 0, "The total number of timesteps must be above 0."
    assert numsaves > 0, "The total number of saves must be above 0."
    assert (timesteps % numsaves) == 0, "The number of saves must be a factor of the number of timesteps."

    assert not no_graphics if observations == 'camera' else True, "No graphics mode is only possible with raycast observations, not camera observations."
    assert resolution >= 4 & resolution <= 512 if observations == 'camera' else True, "Camera observation resolution must be between 4 and 512 inclusive."
    assert ((resolution % 2) == 1) & (
            resolution >= 1) if observations == 'raycast' else True, "Raycast observation resolution must be a positive odd number"

    sb3_algorithm = algorithm_choice(algorithm)

    # Make log directory
    logdir = utils.make_logdir(logdir, task)

    # Find environment
    if env is None:
        env_path = str(utils.find_env_path(Path("./aai/")))
    else:
        env_path = str(env)

    # Construct input space and policy
    if observations == 'raycast':

        res = 4
        camera = False
        raycast = True
        grayscale = False

        if algorithm.lower() == 'recurrent_ppo':
            policy = "MlpLstmPolicy"
        else:
            policy = "MlpPolicy"

    elif observations == 'camera':

        res = resolution
        camera = True
        raycast = False
        grayscale = grayscale

        if algorithm.lower() == 'recurrent_ppo':
            policy = "CnnLstmPolicy"
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                                 enable_critic_lstm=False,
                                 lstm_hidden_size=32, )
        else:
            policy = "CnnPolicy"
            policy_kwargs = dict(activation_fn=th.nn.ReLU)
    else:
        raise ValueError("Choose 'raycast' or 'camera' observations.")

    print(f"{algorithm} agent with {policy}.")

    # Weights & Biases Integration
    if wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        wandb.login(verify=True, key=WANDB_API_KEY)

        config = {
            "policy_type": policy,
            "total_timesteps": timesteps,
            "env_name": "animalai3",
        }

        run = wandb.init(project="sb3-animalai",
                         config=config,
                         sync_tensorboard=True,  # auto-upload tensorboard metrics
                         monitor_gym=False,
                         # auto-upload videos of agents playing game - requires stable_baselines3.common.monitor.Monitor
                         save_code=False,  # upload code running agent
                         )

    print(type(logdir))

    # Create an AnimalAI environment
    env = AnimalAIEnvironment(
        file_name=env_path,
        log_folder=str(logdir),
        arenas_configurations=task,
        base_port=5000 + random.randint(0, 1000),
        resolution=res,
        useCamera=camera,
        useRayCasts=raycast,
        raysPerSide=(res - 1) / 2 if raycast else 2,
        rayMaxDegrees=raycast_degrees if raycast else 60,
        no_graphics=no_graphics,
        grayscale=grayscale,
        timescale=aai_timescale,
        inference=inference,  # change to true to watch agent while training in full screen.
    )
    # Make it compatible with legacy Gym v0.21 API
    env = UnityToGymWrapper(
        env,
        uint8_visual=True,  # CAN UNCOMMENT WHEN R-PPO IS FIXED: if camera else False,
        allow_multiple_obs=False,  # Also provide health, velocity (x, y, z), and global position (x, y, z)
        flatten_branched=True
        # False if algorithm.lower() == 'recurrent_ppo' else True,  # Necessary if the agent doesn't support MultiDiscrete action space.
    )

    print("Starting training...")
    # Will automatically use Shimmy to convert the legacy Gym v0.21 API to the Gymnasium API

    # Initialise agent
    if from_checkpoint is None:
        if sb3_algorithm == "recurrent_ppo":
            model = sb3_algorithm(policy,
                                  env,  # type: ignore
                                  device=device,
                                  policy_kwargs=policy_kwargs,
                                  verbose=1,
                                  tensorboard_log=os.path.join(logdir, f"tensorboard/runs/{run.id}") if wandb else None,
                                  n_steps=512,
                                  batch_size=128,
                                  gamma=0.999,
                                  gae_lambda=0.94,
                                  learning_rate=1e-5
                                  )
        else:
            model = sb3_algorithm(policy,
                                  env,  # type: ignore
                                  device=device,
                                  policy_kwargs=policy_kwargs,
                                  verbose=1,
                                  tensorboard_log=os.path.join(logdir, f"tensorboard/runs/{run.id}") if wandb else None
                                  )
        reset_num_timesteps = True
    else:
        model = sb3_algorithm.load(from_checkpoint)
        model.set_env(env)
        reset_num_timesteps = False

    per_save_steps = timesteps / numsaves

    for saves in range(numsaves):
        before = time.time()
        model.learn(total_timesteps=per_save_steps,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=None if not wandb else WandbCallback(),
                    )
        model.save(os.path.join(logdir, f'training-{(saves + 1) * per_save_steps}'))
        reset_num_timesteps = False
        after = time.time()
        print(f"Total runtime: {(after - before) / 60} minutes for one set of {numsaves} saves.")

    env.close()

    if wandb:
        run.finish()


def main():
    before = time.time()
    num_time_steps = 1 * 10 ** 6
    train(task=Path("../configs/foragingTask/foragingTaskSpawnerTree.yml"),
          algorithm="ppo",
          env=Path("/Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/aai/env/AnimalAI.app"),
          observations='camera',
          timesteps=num_time_steps,
          resolution=64,
          numsaves=10,
          wandb=True,
          from_checkpoint=None,
          aai_timescale=300,
          inference=False,
          logdir=Path("/Users/mgm61/Documents/cambridge_cfi/aai3-paper-experiments/recurrent_ppo/logdir/foraging/foraging-train"),
          )
    after = time.time()
    print(f"Total runtime: {(after - before) / 60} minutes for {num_time_steps} time steps.")


if __name__ == "__main__":
    main()
