from Experiment import Experiment
import numpy as np

class MultiGeneric(Experiment):
    def __init__(self, learners=[], environment=None, **kwargs):
        super(MultiGeneric, self).__init__(**kwargs)
        self.learners = learners
        self.environment = environment

    def reset(self):
        self.environment.reset()
        for l in self.learners:
            l.reset()

    def run_episode(self,trial, episode):
        self.environment.start_setup()
        actions = {}
        terminals = {}
        step = 0
        learners = self.learners
        for learner in learners:
            state = self.environment.start(learner.id)
            action = learner.start(state)
            actions[learner.id] = action
            terminals[learner.id] = False
        while self.in_step_limit(step) and len(learners):
            # only select learners that are not finished
            for learner in np.random.permutation(learners):
                id = learner.id
                state, reward, terminals[id] = self.environment.step(id, actions[id])
                if not terminals[id]:
                    actions[id] = learner.step(reward, state)
                else:
                    learner.end(trial, episode, reward)
            step += 1
            learners = [l for l in self.learners if not terminals[l.id]]