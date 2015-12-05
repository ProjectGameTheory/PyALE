
class Experiment(object):

    def __init__(self, max_steps=None, episodes=1, trials=1):
        self.trials = trials
        self.episodes = episodes
        self.max_steps = max_steps
        self.current_episode = 0
        self.current_trial = 0

    def run_episode(self):
        pass

    def run_trial(self):
        for i in range(self.episodes):
            self.current_episode = i
            self.run_episode()

    def run(self):
        for i in range(self.trials):
            self.current_trial = i
            self.run_trial()
