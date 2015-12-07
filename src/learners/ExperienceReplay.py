from Learner import Learner
import numpy as np


class ExperienceReplay(Learner):

    def __init__(self, replays=1, learner=None, max_samples=None):
        self.replays = replays
        self.learner = learner
        self.sample_database = []
        self.max_samples = max_samples
        self.num_samples = 0
        self.num_steps = 0
        self.tot_reward = 0
        self.trajectory_length = 100

    def start(self, state):
        super(ExperienceReplay, self).start(state)
        return self.learner.start(state)

    def add_sample(self, k, phi, action, state_ns, reward):
        sample = [k, phi, action, state_ns, reward]
        if not self.max_samples or self.num_samples < self.max_samples:
            self.sample_database.append(sample)
        else:
            i = self.num_samples % self.max_samples
            self.sample_database[i] = sample

    def step(self, reward, state):
        # select next action based on current values
        phi_ns = self.learner.features.phi(state)
        action_ns, values = self.learner.select_action(phi_ns)
        # add sample to sample_database
        self.add_sample(self.num_steps, self.learner.phi, self.learner.action, state, reward)
        self.num_samples += 1
        #if self.num_steps == self.l*self.trajectory_length:
        #    self.learn_samples()
        #    self.l += 1
        # update state/action
        self.learner.save_phi_action(phi_ns, action_ns)
        # update step/reward count
        super(ExperienceReplay, self).step(reward, state)
        return action_ns

    def end(self, reward):
        # make a batch update
        self.learn_samples()
        super(ExperienceReplay, self).end(reward)

    def learn_samples(self):
        for i in range(self.replays * self.num_steps):
            j = np.random.randint(0, len(self.sample_database))
            sample = self.sample_database[j]
            # set current state and action fo learner
            self.learner.save_phi_action(sample[1], sample[2])
            # simulate step
            self.learner.step(sample[4], sample[3])
