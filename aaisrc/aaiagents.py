## Agents functions for Animal-AI 3 Paper experiments
## Author: K. Voudouris (C) 2023.

import numpy as np
import random
import warnings

from collections import deque
from scipy.special import softmax

from animalai.envs.actions import AAIActions, AAIAction
from animalai.envs.raycastparser import RayCastParser
from animalai.envs.raycastparser import RayCastObjects

class RandomActionAgent:
    """Implements a random walker with many changeable parameters"""

    def __init__(self, max_step_length = 10, step_length_distribution = 'fixed', norm_mu = 5, norm_sig = 1, beta_alpha = 2, beta_beta = 2, cauchy_mode = 5, gamma_kappa = 9, gamma_theta = 0.5, weibull_alpha = 2, poisson_lambda = 5, action_biases = [1,1,1,1,1,1,1,1,1], prev_step_bias = 0, remove_prev_step = False):
        self.max_step_length = max_step_length 
        self.step_length_distribution = step_length_distribution
        self.norm_mu = norm_mu
        self.norm_sig = norm_sig
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.cauchy_mode = cauchy_mode
        self.gamma_kappa = gamma_kappa
        self.gamma_theta = gamma_theta
        self.weibull_alpha = weibull_alpha
        self.poisson_lambda = poisson_lambda
        self.action_biases = action_biases
        self.prev_step_bias = prev_step_bias
        self.remove_prev_step = remove_prev_step

    def get_num_steps(self, prev_step: int):
        
        if self.step_length_distribution == 'fixed':
            num_steps = self.max_step_length
        
        elif self.step_length_distribution == 'uniform': 
            num_steps = random.randint(0, self.max_step_length)

        elif self.step_length_distribution == 'normal':
            num_steps = -1
            while num_steps <= 0: # to make sure that num_steps is always a natural number
                num_steps = int(np.random.normal(self.norm_mu, self.norm_sig))

        elif self.step_length_distribution == 'beta':
            num_steps = int(np.random.beta(self.beta_alpha, self.beta_beta) * self.max_step_length) #rescale it to be bounded by 0 and max_step_length rather than by 0 and 1

        elif self.step_length_distribution == 'cauchy':
            num_steps = -1
            while num_steps < 0:
                num_steps = int(np.random.standard_cauchy() + self.cauchy_mode)
        
        elif self.step_length_distribution == 'gamma':
            num_steps = -1
            while num_steps < 0:
                num_steps = int(np.random.gamma(self.gamma_kappa, self.gamma_theta))
        
        elif self.step_length_distribution == 'weibull':
            num_steps = int(np.random.weibull(self.weibull_alpha) * self.max_step_length) #rescale it to be bounded by 0 and max_step_length rather than by 0 and 1
        
        elif self.step_length_distribution == 'poisson':
            num_steps = int(np.random.poisson(self.poisson_lambda))
        
        else:
            raise ValueError("Distribution not recognised.")

        if num_steps > 100:
            warning_string = 'The number of steps chosen is: ' + str(num_steps) + '. Try toggling distribution parameters as your agent might get stuck.'
            warnings.warn(warning_string)

        
        step_list = deque([prev_step]*num_steps)
        return step_list

    def get_new_action(self, prev_step: int):

        """
        Provide a vector of 9 real values, one for each action, which is then softmaxed to provide the probability of selecting that action. Relative differences between the values is what is important. 

        Provide an initial probability of selecting the previous step again. If that action is not selected, then the next step is picked according to the softmaxed action biases. The previous action can be removed
        from the softmaxed biases (by continually sampling until an action is picked that is not the previous action), by changing `remove_prev_step` to `True`.
        """

        assert(len(self.action_biases) == 9), "You must provide biases for all nine (9) actions. A uniform distribution is [1,1,1,1,1,1,1,1,1]"

        assert(self.prev_step_bias >= 0 and self.prev_step_bias <= 1), "The bias towards the previous action must be a scalar value between 0 and 1."

        
        action_is_prev_step = np.random.choice(a = [False,True], size = 1, p = [(1-self.prev_step_bias), self.prev_step_bias]) # should the action be the previous step?

        if action_is_prev_step:
            action = prev_step
        else:
            if self.remove_prev_step:
                action_biases_softmax = softmax(self.action_biases)
                action = prev_step
                while action == prev_step:
                    action = np.random.choice(a = [0,1,2,3,4,5,6,7,8], size = 1, p = action_biases_softmax)
            else:
                action_biases_softmax = softmax(self.action_biases)
                action = np.random.choice(a = [0,1,2,3,4,5,6,7,8], size = 1, p = action_biases_softmax)
        
        action = int(action)

        return action
    


class Heuristic():
    """Implements a simple heuristic agent (Braitenberg Vehicle)
    It heads towards good goals and away from bad goals.
    It navigates around immoveable objects directly ahead 
    If it is stationary it turns around
    """
    def __init__(self, no_rays, max_degrees, verbose=False):
        self.verbose = verbose # do you want to see the observations and actions?
        self.no_rays = no_rays # how many rays should the agent have?
        assert(self.no_rays % 2 == 1), "Number of rays must be an odd number." 
        self.max_degrees = max_degrees # how many degrees do you want the rays spread over?
        """
        We specify six types of objects here. This set can be expanded to include more objects if you wish to design further rules.
        """
        self.listOfObjects = [RayCastObjects.ARENA, 
                              RayCastObjects.IMMOVABLE, 
                              RayCastObjects.MOVABLE, 
                              RayCastObjects.GOODGOAL, 
                              RayCastObjects.GOODGOALMULTI, 
                              RayCastObjects.BADGOAL]
        
        self.raycast_parser = RayCastParser(self.listOfObjects, self.no_rays) #initialize a class to parse raycasts for these objects
        self.actions = AAIActions() # initalise the action set
        self.prev_action = self.actions.NOOP # initialise the first action, chosen to be forwards 

    def prettyPrint(self, obs) -> str:
        """Prints the parsed observation"""
        return self.raycast_parser.prettyPrint(obs) #prettyprints the observation in a nice format
    
    def checkStationarity(self, raycast): #checks whether the agent is stationary by examining its velocities. If they are below 1 in all directions, it turns.
        vel_observations = raycast['velocity']
        if self.verbose:
            print("Velocity observations")
            print(vel_observations)
        bool_array = (vel_observations <= np.array([1,1,1]))
        if sum(bool_array) == 3:
            return True
        else:
            return False
    
    def get_action(self, observations) -> AAIAction: #select an action based on observations
        """Returns the action to take given the current parsed raycast observation and other observations"""
        obs = self.raycast_parser.parse(observations)
        if self.verbose:
            print("Raw Raycast Observations:")
            print(obs)
            print("Pretty Raycast Observations:")
            self.raycast_parser.prettyPrint(observations)

        newAction = self.actions.FORWARDS.action_tuple #initialise the new action to be no action

        """
        If the agent is stationary, and it hasn't previously gone forwardsleft or forwardsright, it must be the first step. So choose one of those actions at random (p(0.5))
        """
        if self.checkStationarity(observations) and self.prev_action != self.actions.FORWARDSLEFT and self.prev_action != self.actions.FORWARDSRIGHT:
            select_LR = random.randint(0,1)
            if select_LR == 0:
                newAction = self.actions.FORWARDSLEFT
            else:
                newAction = self.actions.FORWARDSRIGHT
        elif self.checkStationarity(observations) and self.prev_action == self.actions.FORWARDSLEFT: # otherwise if stationary, continue in the same direction (it must be stuck by an obstacle)
            newAction = self.actions.FORWARDSLEFT
        elif self.checkStationarity(observations) and self.prev_action == self.actions.FORWARDSRIGHT:
            newAction = self.actions.FORWARDSRIGHT
        elif self.ahead(obs, RayCastObjects.GOODGOALMULTI) and not self.checkStationarity(observations): #if it's not stationary and it sees a good goal ahead, go forwards
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.GOODGOALMULTI) and not self.checkStationarity(observations): # if it's to the left, rotate left
            newAction = self.actions.LEFT
        elif self.right(obs, RayCastObjects.GOODGOALMULTI) and not self.checkStationarity(observations): # if it's to the right, rotate right
            newAction = self.actions.RIGHT
        elif self.ahead(obs, RayCastObjects.GOODGOAL) and not self.checkStationarity(observations): #as above for good goals
            newAction = self.actions.FORWARDS
        elif self.left(obs, RayCastObjects.GOODGOAL) and not self.checkStationarity(observations):
            newAction = self.actions.LEFT
        elif self.right(obs, RayCastObjects.GOODGOAL) and not self.checkStationarity(observations):
            newAction = self.actions.RIGHT
        elif self.ahead(obs, RayCastObjects.BADGOAL) and not self.checkStationarity(observations): #the opposite for bad goals
            newAction = self.actions.BACKWARDS
        elif self.left(obs, RayCastObjects.BADGOAL) and not self.checkStationarity(observations):
            newAction = self.actions.RIGHT
        elif self.right(obs, RayCastObjects.BADGOAL) and not self.checkStationarity(observations):
            newAction = self.actions.LEFT
        elif self.ahead(obs, RayCastObjects.IMMOVABLE) and not self.checkStationarity(observations): # if there is an obstacle ahead move forwardsleft or forwardsright randomly to start navigating around it
            select_LR = random.randint(0,1)
            if select_LR == 0:
                newAction = self.actions.FORWARDSLEFT
            else:
                newAction = self.actions.FORWARDSRIGHT
        # Otherwise, if there is an obstacle ahead and the previous action was forwardsleft OR if there is an obstacle to the left and nothing ahead and the agent is not stationary, continue forwardsleft to continue navigating around
        elif ((self.ahead(obs, RayCastObjects.IMMOVABLE) and self.prev_action == self.actions.FORWARDSLEFT) or (self.left(obs, RayCastObjects.IMMOVABLE) and not self.ahead(obs, RayCastObjects.IMMOVABLE))) and not self.checkStationarity(observations):
            newAction = self.actions.FORWARDSLEFT
        # vice versa for if the right side. This way the agent can navigate around obstacles
        elif ((self.ahead(obs, RayCastObjects.IMMOVABLE) and self.prev_action == self.actions.FORWARDSRIGHT) or (self.right(obs, RayCastObjects.IMMOVABLE) and not self.ahead(obs, RayCastObjects.IMMOVABLE))) and not self.checkStationarity(observations):
            newAction = self.actions.FORWARDSRIGHT
        else: #otherwise, continue the same action as before
            newAction = self.prev_action        
        self.prev_action = newAction
        
        if self.verbose:
            print("Action selected:")
            print(newAction.name)
        newActionTuple = newAction.action_tuple
        
        return newActionTuple
    
    def ahead(self, obs, object): #returns a true if the object is detected in the middle ray.
        """Returns true if the input object is ahead of the agent"""
        if(obs[self.listOfObjects.index(object)][int((self.no_rays-1)/2)] > 0):
            if self.verbose:
                print("found " + str(object) + " ahead")
            return True
        return False

    def left(self, obs, object): #returns a true if the object is in one of the left rays
        """Returns true if the input object is left of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(object)][i] > 0):
                if self.verbose:
                    print("found " + str(object) + " left")
                return True
        return False

    def right(self, obs, object): #returns a true if the object is in one of the right rays
        """Returns true if the input object is right of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(object)][i+int((self.no_rays-1)/2) + 1] > 0):
                if self.verbose:
                    print("found " + str(object) + " right")
                return True
        return False
    


class buttonHeuristic():
    """Implements a simple heuristic agent (Braitenberg Vehicle)
    It heads towards good goals and away from bad goals.
    It navigates around immoveable objects directly ahead 
    If it is stationary it turns around
    """
    def __init__(self, no_rays, max_degrees, verbose=False):
        self.verbose = verbose # do you want to see the observations and actions?
        self.no_rays = no_rays # how many rays should the agent have?
        assert(self.no_rays % 2 == 1), "Number of rays must be an odd number." 
        self.max_degrees = max_degrees # how many degrees do you want the rays spread over?
        """
        We specify six types of objects here. This set can be expanded to include more objects if you wish to design further rules.
        """
        self.listOfObjects = [RayCastObjects.ARENA, 
                              RayCastObjects.IMMOVABLE, 
                              RayCastObjects.MOVABLE, 
                              RayCastObjects.GOODGOAL, 
                              RayCastObjects.GOODGOALMULTI, 
                              RayCastObjects.BADGOAL,
                              RayCastObjects.PILLARBUTTON]
        
        self.raycast_parser = RayCastParser(self.listOfObjects, self.no_rays) #initialize a class to parse raycasts for these objects
        self.actions = AAIActions() # initalise the action set
        self.prev_action = self.actions.NOOP # initialise the first action, chosen to be forwards 

    def prettyPrint(self, obs) -> str:
        """Prints the parsed observation"""
        return self.raycast_parser.prettyPrint(obs) #prettyprints the observation in a nice format
    
    def checkStationarity(self, raycast): #checks whether the agent is stationary by examining its velocities. If they are below 1 in all directions, it turns.
        vel_observations = raycast['velocity']
        if self.verbose:
            print("Velocity observations")
            print(vel_observations)
        bool_array = (vel_observations <= np.array([0.01,0.01,0.01]))
        if sum(bool_array) == 3:
            return True
        else:
            return False
    
    def get_action(self, observations) -> AAIAction: #select an action based on observations
        """Returns the action to take given the current parsed raycast observation and other observations"""
        obs = self.raycast_parser.parse(observations)
        if self.verbose:
            print("Raw Raycast Observations:")
            print(obs)
            print("Pretty Raycast Observations:")
            self.raycast_parser.prettyPrint(observations)

        newAction = self.actions.NOOP #initialise the new action to be no action

        """
        If the agent is stationary, and it hasn't previously gone forwardsleft or forwardsright, it must be the first step. So choose one of those actions at random (p(0.5))
        """
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
            else:
                newAction = self.prev_action
        elif self.checkStationarity(observations) and (self.prev_action != self.actions.LEFT or self.prev_action != self.actions.RIGHT):
            
            if self.prev_action == self.actions.NOOP:
                select_action = random.randint(0,5)
                if select_action == 0:
                    newAction = self.actions.FORWARDS
                elif select_action == 1:
                    newAction = self.actions.FORWARDSLEFT
                elif select_action == 2:
                    newAction = self.actions.FORWARDSRIGHT
                elif select_action == 3:
                    newAction = self.actions.BACKWARDS
                elif select_action == 4:
                    newAction = self.actions.BACKWARDSLEFT
                else:
                    newAction = self.actions.BACKWARDSRIGHT
            else:
                newAction = self.actions.NOOP
               
        self.prev_action = newAction
        
        if self.verbose:
            print("Action selected:")
            print(newAction.name)

        newActionTuple = newAction.action_tuple
        
        return newActionTuple
    
    def ahead(self, obs, object): #returns a true if the object is detected in the middle ray.
        """Returns true if the input object is ahead of the agent"""
        if(obs[self.listOfObjects.index(object)][int((self.no_rays-1)/2)] > 0):
            if self.verbose:
                print("found " + str(object) + " ahead")
            return True
        return False

    def left(self, obs, object): #returns a true if the object is in one of the left rays
        """Returns true if the input object is left of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(object)][i] > 0):
                if self.verbose:
                    print("found " + str(object) + " left")
                return True
        return False

    def right(self, obs, object): #returns a true if the object is in one of the right rays
        """Returns true if the input object is right of the agent"""
        for i in range(int((self.no_rays-1)/2)):
            if(obs[self.listOfObjects.index(object)][i+int((self.no_rays-1)/2) + 1] > 0):
                if self.verbose:
                    print("found " + str(object) + " right")
                return True
        return False