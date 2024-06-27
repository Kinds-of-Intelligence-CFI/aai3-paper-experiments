import random
from pathlib import Path

import numpy as np
import pandas as pd

from animalai import AnimalAIEnvironment # type: ignore
from animalai.actions import AAIActions # type: ignore
from animalai.agents.randomActionAgent import RandomActionAgent # type: ignore
from animalai.executable import find_executable # type: ignore

def run_evaluation(config: str, agent: RandomActionAgent, num_episodes: int, timescale: int = 1):
    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path(".."))),
        arenas_configurations=config,
        base_port=5005 + random.randint(0, 1000),
        useRayCasts=True,
        rayMaxDegrees=30,
        raysPerSide=1,
        play=False,
        inference=False,
        timescale=timescale,
        no_graphics=False,
    )

    behavior = list(env.behavior_specs.keys())[0]

    actions = AAIActions()

    episode_rewards = []

    # Run episodes
    for _episode in range(num_episodes):
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
                    print(f"Episode Reward: {episodeReward}")
                    episode_rewards.append(episodeReward)
                    done = True
                    break  # Go to next episode
                
            new_action = agent.get_new_action(prev_step=previous_action)

    print("Closing environment")
    env.close()
    print("Environment Closed")

    results_dataframe = pd.DataFrame({"EpisodeNumber" : [x for x in range(num_episodes)],
                                     "FinalReward" : episode_rewards})

    return results_dataframe

def main():
    num_runs = 100
    agent = RandomActionAgent(step_length_distribution=lambda: np.random.normal(
        5, 1))  # a Rayleigh walker (sampling from normal distribution)
    
    print(f"Running {num_runs} of Heuristic Agent on foraging task.")
    config_file = "../configs/foragingTask/foragingTaskSpawnerTree.yml"
    results = run_evaluation(config_file, agent, num_runs)
    results.to_csv("./results/foragingTask/heuristic.csv", index=False)

    print(f"Running {num_runs} of bespoke Heuristic Agent on operant chamber task.")
    config_file = "../configs/operantChamberTask/operantChamberTask.yml"
    results = run_evaluation(config_file, agent, num_runs)
    results.to_csv("./results/operantChamberTask/heuristic.csv", index=False)

    print(f"Running {num_runs} of bespoke Heuristic Agent on what-where-when task.")
    config_file = "../configs/whatWhereWhenTask/whatWhereWhenTask.yml"
    results = run_evaluation(config_file, agent, num_runs)
    results.to_csv("./results/whatWhereWhenTask/heuristic.csv", index=False)

    print("Finished simulation.")
