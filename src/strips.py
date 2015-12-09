from learners.Q import Q
from learners.RewardShaping import RewardShaping
from shaping.STRIPS import STRIPS
from policies.EGreedy import EGreedy
from traces.Eligibility import Eligibility
from features.Feature import Feature
from environments.GridWorld import GridWorld
from experiments.Generic import Generic
from experiments.MultiGeneric import MultiGeneric
from environments.RoomWorld import RoomWorld
import numpy as np

from monitors.Event import episode_ended
from monitors.Printer import Printer
from monitors.Writer import Writer

from strips_plans import *
from ast import literal_eval

actions = [0, 1, 2, 3]
grid = []

with open("grid.txt") as f:
    for line in f:
        grid.insert(0,literal_eval(line[:-1])) #Opposite order
grid = np.array(grid)

#monitors
printer = Printer()
printer.follow(episode_ended)
printer.activate()
writer = Writer()
writer.follow(episode_ended)
writer.activate()


nr_of_agents = 2
environment = RoomWorld(grid, population_size=nr_of_agents, ids=np.arange(nr_of_agents),begins=[[7,4], [7,14]],goal=[1,1])

def state_to_strips(env, id, state):
    strips_state = set([])
    strips_state.add('in_' + grid[state][2])
    for flag in env.flags_taken[id]:
        strips_state.add('taken_flag' + flag)
    return strips_state

shaper = STRIPS(strips_plan=strips_plans_individual[0], convert=state_to_strips, omega=600/9)

#agent
e_greedy = EGreedy(epsilon=0.2)
grid_size = np.shape(grid)
no_features = Feature(state_length= grid_size[0]*grid_size[1])
trace = Eligibility(lambda_=0.9, actions=actions, shape=(no_features.num_features(), len(actions)))
q = Q(actions=actions, alpha=0.1, gamma=0.9, policy=e_greedy, features=no_features, trace=trace)
learners = [RewardShaping(learner=q, shaper=shaper) for i in range(nr_of_agents)]

experiment = MultiGeneric(max_steps=None, episodes=5000, trials=1, learners=learners, environment=environment)

#run generic experiment
experiment.run()
writer.save()