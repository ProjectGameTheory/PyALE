from Experiment import Experiment
import numpy as np

class MultiGeneric(Experiment):
    def __init__(self, learners=[], environment=None, **kwargs):
        super(MultiGeneric, self).__init__(**kwargs)
        self.learners = learners
        self.environment = environment

    def run_episode(self):
        self.environment.start_setup()
        actions = {}
        terminals = {}
        step = 0
        learners = np.random.permutation(self.learners)
        for learner in learners:
            state = self.environment.start(learner.id)
            action = learner.start(state)
            actions[learner.id] = action
            terminals[learner.id] = False
        while self.in_step_limit(step) and len(learners):
            learners = np.random.permutation(self.learners)
            # only select learners that are not finished
            learners = [l for l in learners if l.id not in terminals or not terminals[l.id]]
            for learner in learners:
                id = learner.id
                state, reward, terminals[id] = self.environment.step(id, actions[id])
                if not terminals[id]:
                    actions[id] = learner.step(reward, state)
                else:
                    learner.end(reward)
            step += 1

