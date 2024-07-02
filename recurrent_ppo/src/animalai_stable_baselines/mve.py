from __future__ import annotations

import csv
import os
import random
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from numpy.typing import NDArray

import numpy as np
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from sb3_contrib import RecurrentPPO
from src.animalai_stable_baselines.yaml_handling import find_yaml_files, yaml_combinor

def mve(model_save_path: str,
        load: BaseAlgorithm.load) -> None:
    model = load(model_save_path)
    for _ in range(1000):
#        random_obs = np.random.randint(low=0, high=1, size=(64, 64, 3))
        random_obs = np.random.rand(64, 64, 3)
        action, _ = model.predict(random_obs, deterministic=False,)
        print(action.item())

def example():
    mve(model_save_path="logdir/competition-curriculum/competition-curriculum-timescale300-L1_10_5M/training-5000000.0", load=RecurrentPPO.load)

if __name__ == "__main__":
    example()
