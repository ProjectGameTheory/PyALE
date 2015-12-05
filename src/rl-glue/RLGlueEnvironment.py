from rlglue.environment.Environment import Environment
from rlglue.types import Observation
from rlglue.types import Reward_observation_terminal


class RLGlueEnvironment(Environment):

    def __init__(self, environment=None):
        self.environment = environment

    def create_observation(self, state):
        observation = Observation()
        observation.intArray = state.tolist()
        return observation

    def get_action(self, action):
        return action.intArray[0]

    def env_init(self):
        spec = "VERSION RL-Glue-2.02 PROBLEMTYPE episodic DISCOUNTFACTOR 0.9 " +\
               "OBSERVATIONS INTS (0 1) 100" +\
               "ACTIONS INTS (0 3)  REWARDS (0. 10.)"
        return spec

    def env_start(self):
        state = self.environment.start()
        return self.create_observation(state)

    def env_step(self, action):
        state, reward, terminal = self.environment.step(self.get_action(action))

        rot = Reward_observation_terminal()
        rot.r = reward
        rot.o = self.create_observation(state)
        rot.terminal = terminal
        return rot

    def env_cleanup(self):
        pass
