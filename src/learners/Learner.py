import numpy as np
from monitors.Event import episode_ended

class Learner(object):
    '''
    Only works with binary features (non-sparse)
    '''

    def __init__(self, actions=[], alpha=0.1, gamma=0.999,
                 policy=None, trace=None, features=None, normalization=False):
        self.actions = np.array(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.trace = trace
        self.features = features
        self.theta = np.zeros(
            (self.features.num_features(), self.num_actions()))
        # current state
        self.phi = None
        # last taken action
        self.action = None
        self.normalization = normalization
        # used for normalization
        self.base_reward = None
        self.num_steps = 0
        self.tot_reward = 0

    def num_actions(self):
        return len(self.actions)

    def dim_action(self):
        assert len(
            self.actions) > 0, 'Must be able to execute at least one action'
        return self.actions[0].size

    '''
    Reward normalization. Stores first nonzero reward and divides all future
    rewards by its abolute value.  
    '''

    def normalize_reward(self, reward):
        if reward == 0.:
            return reward
        if self.base_reward is None:
            self.base_reward = np.abs(reward)
        return reward / self.base_reward

    def normalize_alpha(self, phi):
        # normalize alpha with nr of active features
        return self.alpha / float(np.sum(phi != 0.))

    def start(self, state):
        self.trace.reset()
        self.num_steps = 0
        self.tot_reward = 0

    def step(self, reward, state):
        self.num_steps += 1
        self.tot_reward += reward

    def end(self, reward):
        self.num_steps += 1
        self.tot_reward += reward
        episode_ended.notify(self.num_steps, self.tot_reward)

    def get_value(self, phi, action):
        action_idx = np.where(self.actions == action)[0][0]
        return np.dot(phi, self.theta[:, action_idx])
        # non-sparse update
        # return np.sum(self.theta[phi,action_idx])

    def get_all_values(self, phi):
        return np.dot(phi, self.theta)
        # non-sparse update
        # return np.sum(self.theta[phi,:],axis=0)

    def save_phi_action(self, phi, action):
        self.phi = phi
        self.action = action
