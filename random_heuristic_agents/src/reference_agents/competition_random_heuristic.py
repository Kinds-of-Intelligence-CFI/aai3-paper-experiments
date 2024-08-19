## Using AAI v3.1.3

import csv
import fnmatch
import random
import os
from pathlib import Path

import numpy as np

from animalai import AnimalAIEnvironment # type: ignore
from animalai.actions import AAIActions # type: ignore
from animalai.agents.braitenberg import Braitenberg # type: ignore
from animalai.executable import find_executable # type: ignore
from animalai.agents.randomActionAgent import RandomActionAgent # type: ignore

def find_yaml_files(directory):
    yaml_files = []
    task_names = []
    
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.yml') + fnmatch.filter(filenames, '*.yaml'):
            yaml_files.append(os.path.join(root, filename))
            task_names.append(filename)
    
    return yaml_files, task_names

def run_evaluation(config_folder: str, agent, agent_type: str, save_path_csv: str, timescale: int = 1, degrees = 30):

    paths, names = find_yaml_files(config_folder)

    port = 1000

    for p, n in zip(paths, names):
        print(f"Running episode: {n}")
        if agent_type == "Random":
            env = AnimalAIEnvironment(
                file_name=str(find_executable(Path(".."))),
                arenas_configurations=str(p),
                base_port=port,
                worker_id=0,
                useRayCasts=True,
                rayMaxDegrees=degrees,
                raysPerSide=1,
                play=False,
                inference=False,
                timescale=timescale,
                no_graphics=False,
                )

            behavior = list(env.behavior_specs.keys())[0]
            print("Running Random Agent...")
            actions = AAIActions()

            env.step()
            done = False
            episodeReward = 0

            previous_action = actions.NOOP
            new_action = agent.get_new_action(prev_step=previous_action)

            while not done:
                step_list = agent.get_num_steps(prev_step=new_action)

                for action in step_list:
                    env.set_actions(behavior, action.action_tuple)
                    env.step()
                    previous_action = action
                
                    dec, term = env.get_steps(behavior)

                    # Episode is over
                    if len(term) > 0:
                        episodeReward += term.reward
                        print(f"Episode Reward: {episodeReward} on task: {n}.")
                        done = True
                        break  # Go to next episode
                    
                    episodeReward += dec.reward if len(dec.reward) > 0 else 0
                new_action = agent.get_new_action(prev_step=previous_action)

            print("Closing environment")
            env.close()
            print("Environment Closed")
        
        elif agent_type == "Heuristic":
            print("Running Heuristic Agent...")
            env = AnimalAIEnvironment(
                file_name=str(find_executable(Path(".."))),
                arenas_configurations=str(p),
                base_port=port,
                worker_id=1000,
                useRayCasts=True,
                rayMaxDegrees=degrees,
                raysPerSide=(agent.no_rays - 1) // 2,
                play=False,
                inference=False,
                timescale=timescale,
                no_graphics=False,
                )

            behavior = list(env.behavior_specs.keys())[0]

            episodeReward = 0
            while True:
                # Step the environment
                env.step()
                step, term = env.get_steps(behavior)

                # Episode is over
                if len(term) > 0:
                    episodeReward += term.reward
                    print(f"Episode Reward: {episodeReward}")
                    break  # Go to next episode

                # Get observations
                observations = env.get_obs_dict(step.obs)
                episodeReward += step.reward if len(step.reward) > 0 else 0

                # Get agent action
                action = agent.get_action(observations["rays"])
            
                env.set_actions(behavior, action.action_tuple)

            print("Closing environment")
            env.close()
            print("Environment Closed")
            port += 1
        else:
            raise ValueError(f"Agent type {agent_type} not recognised.")

        
        file_exists = os.path.isfile(save_path_csv)
        print(f"Writing episode score {episodeReward} for episode {n} to {save_path_csv}")
        with open(save_path_csv, 'a' if file_exists else 'w', newline='') as csv_file:
            csv_write = csv.writer(csv_file)
            if not file_exists:
                csv_write.writerow(['episode', 'finalReward'])
            csv_write.writerow([str(n), str(episodeReward)])
        
    return True

def main():
    random.seed(2024)
    folder = "../configs/competition/"
    randomAgent = RandomActionAgent(step_length_distribution=lambda: np.random.normal(
        5, 1))
    heuristicAgent = Braitenberg(no_rays=15)

    simulate = run_evaluation(folder, randomAgent, "Random", "results/competition/randomAgent.csv")
    if simulate:
        print("Random Agent successfully simulated.")
    else:
        print("Error")
    
    simulate = run_evaluation(folder, heuristicAgent, "Heuristic", "results/competition/heuristicAgent.csv", degrees=30)
    if simulate:
        print("Heuristic Agent successfully simulated.")
    else:
        print("Error")
