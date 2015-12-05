from experiments.Experiment import Experiment
import rlglue.RLGlue as RLGlue
import csv


class RLGlueExperiment(Experiment):

    def __init__(self, **kwargs):
        super(RLGlueExperiment, self).__init__(**kwargs)
        if self.max_steps is None:
            self.max_steps = 0

    def run_episode(self):
        terminal = RLGlue.RL_episode(self.max_steps)
        steps = RLGlue.RL_num_steps()
        reward = RLGlue.RL_return()

    def run(self):
        task_spec = RLGlue.RL_init()
        super(RLGlueExperiment, self).run()
        RLGlue.RL_cleanup()
