import random
from pathlib import Path

import numpy as np
import pandas as pd

from animalai import AnimalAIEnvironment # type: ignore
from animalai.actions import AAIAction # type: ignore
from animalai.agents.braitenberg import Braitenberg # type: ignore
from animalai.executable import find_executable # type: ignore
from animalai.raycastparser import RayCastObjects, RayCastParser # type: ignore

def run_evaluation(config: str, agent, num_episodes: int, num_degrees: int = 30, timescale: int = 1, multi_obs: bool = False):
    env = AnimalAIEnvironment(
        file_name=str(find_executable(Path(".."))),
        arenas_configurations=config,
        base_port=5005 + random.randint(0, 1000),
        useRayCasts=True,
        rayMaxDegrees=num_degrees,
        raysPerSide=(agent.no_rays - 1) // 2,
        play=False,
        inference=False,
        timescale=timescale,
        no_graphics=False,
    )

    behavior = list(env.behavior_specs.keys())[0]

    episode_rewards = []

    # Run episodes
    for _episode in range(num_episodes):
        episodeReward = 0
        while True:
            # Step the environment
            env.step()
            step, term = env.get_steps(behavior)

            # Episode is over
            if len(term) > 0:
                episodeReward += term.reward
                print(f"Episode Reward: {episodeReward}")
                episode_rewards.append(episodeReward)
                break  # Go to next episode

            # Get observations
            observations = env.get_obs_dict(step.obs)
            episodeReward += step.reward if len(step.reward) > 0 else 0

            # Get agent action
            if multi_obs:
                action = agent.get_action(observations)
            else:
                action = agent.get_action(observations["rays"])
            
            env.set_actions(behavior, action.action_tuple)

    print("Closing environment")
    env.close()
    print("Environment Closed")

    results_dataframe = pd.DataFrame({"EpisodeNumber" : [x for x in range(num_episodes)],
                                     "FinalReward" : episode_rewards})

    return results_dataframe

class operantBraitenberg(Braitenberg):
  def __init__(self, no_rays):
    super().__init__(no_rays)
    self.listOfObjects = [RayCastObjects.GOODGOAL, RayCastObjects.GOODGOALMULTI, RayCastObjects.BADGOAL, RayCastObjects.IMMOVABLE, RayCastObjects.MOVABLE, RayCastObjects.PILLARBUTTON]
    self.raycast_parser = RayCastParser(self.listOfObjects, self.no_rays) 

  def checkStationarity(self, raycast): #checks whether the agent is stationary by examining its velocities. If they are below 1 in all directions, it turns.
    vel_observations = raycast['velocity']
    bool_array = (vel_observations <= np.array([1,1,1]))
    if sum(bool_array) == 3:
        return True
    else:
        return False

  def get_action(self, observations) -> AAIAction:
    """Returns the action to take given the current parsed raycast observation"""
    obs = self.raycast_parser.parse(observations)

    if not self.checkStationarity(observations):
        if self.ahead(obs, RayCastObjects.GOODGOAL):
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.GOODGOAL):
            newAction = self.actions.LEFT
        elif self.right(obs, RayCastObjects.GOODGOAL):
            newAction = self.actions.RIGHT
        elif self.ahead(obs, RayCastObjects.PILLARBUTTON):
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.PILLARBUTTON):
            newAction = self.actions.LEFT
        elif self.right(obs, RayCastObjects.PILLARBUTTON):
            newAction = self.actions.RIGHT
        elif self.ahead(obs, RayCastObjects.IMMOVABLE) and self.left(obs, RayCastObjects.PILLARBUTTON):
            newAction = self.actions.LEFT
        elif self.ahead(obs, RayCastObjects.IMMOVABLE) and self.right(obs, RayCastObjects.PILLARBUTTON):
            newAction = self.actions.RIGHT
        elif self.ahead(obs, RayCastObjects.IMMOVABLE):
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.IMMOVABLE):
            newAction = self.actions.FORWARDSLEFT
        elif self.right(obs, RayCastObjects.IMMOVABLE):
            newAction = self.actions.FORWARDSRIGHT
        else:
            newAction = self.prev_action
    # elif self.checkStationarity(observations) and self.prev_action == self.actions.LEFT:
    #     newAction = self.actions.FORWARDSLEFT
    else:
        #newAction = self.actions.LEFT
        newAction = self.actions.FORWARDSLEFT
               
    self.prev_action = newAction
        
    return newAction


class ramBraitenberg(Braitenberg):
  def __init__(self, no_rays):
    super().__init__(no_rays)
    self.listOfObjects = [RayCastObjects.GOODGOAL, RayCastObjects.GOODGOALMULTI, RayCastObjects.BADGOAL, RayCastObjects.IMMOVABLE]
    self.raycast_parser = RayCastParser(self.listOfObjects, self.no_rays) 

  def checkStationarity(self, raycast): #checks whether the agent is stationary by examining its velocities. If they are below 1 in all directions, it turns.
    vel_observations = raycast['velocity']
    bool_array = (vel_observations <= np.array([1,1,1]))
    if sum(bool_array) == 3:
        return True
    else:
        return False

  def get_action(self, observations) -> AAIAction:
    """Returns the action to take given the current parsed raycast observation"""
    obs = self.raycast_parser.parse(observations)

    if self.checkStationarity(observations):
        if self.prev_action == self.actions.FORWARDSLEFT:
            newAction = self.actions.LEFT
        else:
            newAction = self.actions.FORWARDSLEFT
    else:
        if self.ahead(obs, RayCastObjects.GOODGOAL) or self.ahead(obs, RayCastObjects.GOODGOALMULTI):
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.GOODGOAL) or self.left(obs, RayCastObjects.GOODGOALMULTI):
            newAction = self.actions.LEFT
        elif self.right(obs, RayCastObjects.GOODGOAL) or self.right(obs, RayCastObjects.GOODGOALMULTI):
            newAction = self.actions.RIGHT
        else:
            newAction = self.actions.FORWARDS

    
               
    self.prev_action = newAction
        
    return newAction


def main():
    num_runs = 100
    
    print(f"Running {num_runs} of Heuristic Agent on foraging task.")
    agent = Braitenberg(no_rays=15)
    config_file = "../configs/foragingTask/foragingTaskSpawnerTree.yml"
    results = run_evaluation(config_file, agent, num_runs, 30, multi_obs=False)
    results.to_csv("./results/foragingTask/heuristic.csv", index=False)

    print(f"Running {num_runs} of bespoke Heuristic Agent on operant chamber task.")
    agent = operantBraitenberg(no_rays=99)
    config_file = "../configs/operantChamberTask/operantChamberTask.yml"
    results = run_evaluation(config_file, agent, num_runs, 30, multi_obs=True)
    results.to_csv("./results/operantChamberTask/heuristic.csv", index=False)

    print(f"Running {num_runs} of bespoke Heuristic Agent on what-where-when task.")
    agent = ramBraitenberg(no_rays=15)
    config_file = "../configs/whatWhereWhenTask/whatWhereWhenTask.yml"
    results = run_evaluation(config_file, agent, num_runs, 30, multi_obs=True)
    results.to_csv("./results/whatWhereWhenTask/heuristic.csv", index=False)

    print("Finished simulation.")
