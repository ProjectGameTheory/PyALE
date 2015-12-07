from monitors.Event import episode_ended


class Learner(object):
    '''
    Only works with binary features (non-sparse)
    '''

    def __init__(self):
        self.num_steps = 0
        self.tot_reward = 0

    def start(self, state):
        self.num_steps = 0
        self.tot_reward = 0

    def step(self, reward, state):
        self.num_steps += 1
        self.tot_reward += reward

    def end(self, reward):
        self.num_steps += 1
        self.tot_reward += reward
        episode_ended.notify(self.num_steps, self.tot_reward)

    def save_phi_action(self, phi, action):
        self.phi = phi
        self.action = action
