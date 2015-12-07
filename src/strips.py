from learners.Q import Q
from learners.RewardShaping import RewardShaping
from shaping.STRIPS import STRIPS
from policies.EGreedy import EGreedy
from traces.Eligibility import Eligibility
from features.Feature import Feature
from environments.GridWorld import GridWorld
from experiments.Generic import Generic
import numpy as np

from monitors.Event import episode_ended
from monitors.Printer import Printer
from monitors.Writer import Writer

actions = [0, 1, 2, 3]
grid = [10, 10]

#monitors
printer = Printer()
printer.follow(episode_ended)
printer.activate()
writer = Writer()
writer.follow(episode_ended)
writer.activate()

# Agent needs to pass by a specific position to get a positive reward
class GridWorldPassBy(GridWorld):
    def __init__(self, pass_by=[5,0], **kwargs):
        super(GridWorldPassBy, self).__init__(**kwargs)
        self.pass_by = pass_by

    def to_binary(self, position, passed):
        state = GridWorld.to_binary(self, position)
        passed = 1 if passed else 0
        return np.append(state, passed)

    def start(self):
        self.current_state = [self.begin, False]
        return self.to_binary(self.begin, False)

    def reward(self, position, passed):
        reward = 0
        if self.terminal(position) and passed:
            reward = 10
        return reward

    def step(self, action):
        position = self.move(self.current_state[0], action)
        passed = True if np.array_equal(position, self.pass_by) else self.current_state[1]
        self.current_state = [position, passed]
        reward = self.reward(position, passed)
        terminal = self.terminal(position)
        return self.to_binary(position, passed), reward, terminal


#environment
start = [0,4]
by = [4,0]
end = [9,4]
environment = GridWorldPassBy(size=grid, begin=start, goal=end, pass_by=by)

#strips plan
strips_plan = [set(['At_start']), set(['At_by', 'passed']), set(['At_goal', 'passed'])]

def state_to_strips(state):
    l = len(state)
    strips_state = set([])
    if state[l-1]:
        strips_state.add('passed')
    position = np.where(state[:l-1] == 1)[0][0]
    if start[0]+start[1]*grid[0] == position:
        strips_state.add('At_start')
    if by[0]+by[1]*grid[0] == position:
        strips_state.add('At_by')
    if end[0]+end[1]*grid[0] == position:
        strips_state.add('At_goal')
    return strips_state

shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=10/3)

#agent
e_greedy = EGreedy(epsilon=0.2)
no_features = Feature(state_length= grid[0]*grid[1]+1)
trace = Eligibility(lambda_=0.9, actions=actions, shape=(no_features.num_features(), len(actions)))
q = Q(actions=actions, alpha=0.1, gamma=0.9, policy=e_greedy, features=no_features, trace=trace)
learner = RewardShaping(learner=q, shaper=shaper)

experiment = Generic(max_steps=None, episodes=5000, trials=1, learner=learner, environment=environment)

#run generic experiment
experiment.run()
writer.save()