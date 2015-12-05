from rlglue.agent.Agent import Agent
from rlglue.types import Action
import numpy as np


class RLGlueAgent(Agent):

    '''
    only supports actions as ints
    only supports observations as ints
    '''

    def __init__(self, learner=None, observation_limits=[0, 0]):
        self.learner = learner
        self.observation_limits = observation_limits

    def state(self, observation):
        mi = self.observation_limits[0]
        ma = self.observation_limits[1]
        return observation.intArray[mi:ma]

    def create_action(self, action):
        if np.isscalar(action):
            action = np.array([action])
        return_action = Action()
        return_action.intArray = [
            action[:self.learner.dim_action()].astype(int)]
        return return_action

    def agent_init(self, taskspec):
        pass

    def agent_start(self, observation):
        state = self.state(observation)
        action = self.learner.start(state)
        return self.create_action(action)

    def agent_step(self, reward, observation):
        state = self.state(observation)
        action = self.learner.step(reward, state)
        return self.create_action(action)

    def agent_end(self, reward):
        self.learner.end(reward)

    def agent_cleanup(self):
        pass