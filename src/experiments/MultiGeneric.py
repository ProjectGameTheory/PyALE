from Experiment import Experiment
import numpy as np
import csv

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
        steps = {}
        ep = 900
        grid_size = [13,18]
        actions = {}
        terminals = {}
        step = 0
        learners = self.learners
        for learner in learners:
            if episode >= ep:
                steps[learner.id] = []
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
                    actions[id] = learner.step(reward, state, episode)
                else:
                    learner.end(trial, episode, reward)
                    if episode >= ep:
                        filename = 'steps-e-' + str(episode) + '-l-' + str(learner.id) + '.csv'
                        with open(filename, 'wb') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',',
                                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(['y', 'x'])
                            writer.writerows(steps[learner.id])
                if episode >= ep:
                    agent_idx = np.where(state == 1)[0][0]
                    pos_idx = agent_idx % (grid_size[0]*grid_size[1])
                    pos = [pos_idx % grid_size[0], pos_idx / grid_size[0]]
                    steps[learner.id].append(pos)
            step += 1
            learners = [l for l in self.learners if not terminals[l.id]]
        # not all episodes terminal, but over step_limit
        if len(learners):
            for learner in learners:
                learner.end(trial, episode, reward)
                if episode >= ep:
                    filename = 'steps-e-' + str(episode) + '-l-' + str(learner.id) + '.csv'
                    with open(filename, 'wb') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['y', 'x'])
                        writer.writerows(steps[learner.id])