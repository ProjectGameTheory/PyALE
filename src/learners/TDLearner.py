from Learner import Learner
import numpy as np


class TDLearner(Learner):

    def __init__(self, actions=[], alpha=0.1, gamma=0.999, policy=None, trace=None, features=None, normalization=False):
        super(TDLearner, self).__init__()
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

    def get_value(self, phi, action):
        action_idx = np.where(self.actions == action)[0][0]
        return np.dot(phi, self.theta[:, action_idx])
        # non-sparse update
        # return np.sum(self.theta[phi,action_idx])

    def get_all_values(self, phi):
        return np.dot(phi, self.theta)
        # non-sparse update
        # return np.sum(self.theta[phi,:],axis=0)

    def num_actions(self):
        return len(self.actions)

    def dim_action(self):
        assert len(
            self.actions) > 0, 'Must be able to execute at least one action'
        return self.actions[0].size

    def idx_action(self, action):
        return np.where(self.actions == action)[0][0]

    def max_actions_idx(self, values_ns):
        return np.arange(values_ns.size)[values_ns == np.max(values_ns)]

    def select_action(self, phi):
        values = self.get_all_values(phi)
        action = self.policy.select_action(actions=self.actions, values=values)
        return action, values

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

    def get_reward(self, reward):
        if self.normalization:
            reward = self.normalize_reward(reward)
        return reward

    def normalize_alpha(self, phi):
        # normalize alpha with nr of active features
        return self.alpha / float(np.sum(phi != 0.))

    def get_alpha(self):
        alpha = self.alpha
        if self.normalization:
            alpha = self.normalize_alpha(alpha)
        return alpha

    def update_theta(self, alpha, reward, phi_ns):
        raise NotImplementedError()

    def start(self, state):
        super(TDLearner, self).start(state)
        self.trace.reset()
        phi_ns = self.features.phi(state)
        action_ns, values_ns = self.select_action(phi_ns)
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return action_ns

    def step(self, reward, state):
        super(TDLearner, self).step(reward, state)
        alpha = self.get_alpha()
        reward = self.get_reward(reward)
        # features from state
        phi_ns = self.features.phi(state)

        self.trace.update(self.phi, self.action, self.gamma)
        # td update
        self.theta += self.update_theta(alpha, reward, phi_ns)
        # action was updated
        return self.action

    def end(self, reward):
        super(TDLearner, self).end(reward)
        alpha = self.get_alpha()
        reward = self.get_reward(reward)

        self.trace.update(self.phi, self.action, self.gamma)
        #reward = self.normalize_reward(reward)
        delta = reward - self.get_value(self.phi, self.action)
        # update weights
        self.theta += alpha * delta * self.trace.values
