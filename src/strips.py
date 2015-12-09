from learners.Sarsa import Sarsa
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
grid_size = np.shape(grid)
#monitors
printer = Printer()
printer.follow(episode_ended)
printer.activate()
writer = Writer()
writer.follow(episode_ended)
writer.activate()


flags = []
for row in grid:
    for pos in row:
        if pos[1]:
            flags.append(pos[1])
pos_length = len(bin(grid_size[0]*grid_size[1]-1)[2:])
num_flags = len(flags) 
def state_to_strips(state):
    # first bits are agent's position (binary encoded)
    pos_idx = 0
    for bit in state[:pos_length]:
        pos_idx = (pos_idx << 1) | bit
    pos = [pos_idx % grid_size[0], pos_idx / grid_size[0]]
    # last bits are agent's flags
    taken = state[-num_flags:]
    strips_state = set([])
    for i in range(num_flags):
        if taken[i]:
            strips_state.add('taken_flag' + flags[i])
    strips_state.add('in_' + grid[pos[0]][pos[1]][2])
    return strips_state

nr_of_agents = 2
learners = []
#begins = [[7,4]]
begins = [[7,4], [7,14]]
#agents
for i in range(nr_of_agents):
    shaper = STRIPS(strips_plan=strips_plans_individual[i], convert=state_to_strips, omega=600/9)
    #agent
    e_greedy = EGreedy(epsilon=0.2)
    no_features = Feature(state_length= pos_length*nr_of_agents + len(flags)*2)
    trace = Eligibility(lambda_=0.9, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.9, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])

experiment = MultiGeneric(max_steps=None, episodes=5000, trials=1, learners=learners, environment=environment)

#run generic experiment
experiment.run()
writer.save()