from Experiment import Experiment

class Generic(Experiment):
    def __init__(self, learner=None, environment=None, **kwargs):
        super(Generic, self).__init__(**kwargs)
        self.learner = learner
        self.environment = environment

    def in_step_limit(self, step):
        return not self.max_steps or step < self.max_steps

    def run_episode(self):
        state = self.environment.start()
        action = self.learner.start(state)
        terminal = False
        step = 0
        while self.in_step_limit(step):
            state, reward, terminal = self.environment.step(action)
            if terminal:
                break
            action = self.learner.step(reward, state)
        self.learner.end(reward)
