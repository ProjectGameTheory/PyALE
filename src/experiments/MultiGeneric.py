from Experiment import Experiment
import numpy as np

class MultiGeneric(Experiment):
    def __init__(self, learners=[], environment=None, **kwargs):
        super(MultiGeneric, self).__init__(**kwargs)
        self.learners = learners
        self.environment = environment

    def run_episode(self):
        learners = np.random.permutation(self.learners)
        actions = {}
        terminals = {}
        step = 0
        for learner in self.learners:
            state = self.environment.start(id)
            action = learner.start(state)
            actions[learner.id] = action
            terminals[id] = False
        while self.in_step_limit(step):
            learners = np.random.permutation(self.learners)
            # only select learners that are not finished
            learners = learners[np.where([terminals[l.id] for l in learners])[0]]
            for learner in learners:
                id = learner.id
                state, reward, terminals[id] = self.environment.step(id, actions[id])
                if not terminals[id]:
                    actions[id] = learner.step(reward, state)
                else:
                    learner.end(reward)
            step += 1
            

