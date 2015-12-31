from learners.Sarsa import Sarsa
from learners.RewardShaping import RewardShaping
from shaping.STRIPS import STRIPS
from shaping.FlagShaper import FlagShaper
from shaping.CombinedShaper import CombinedShaper
from policies.EGreedy import EGreedy
from policies.TimeEGreedy import TimeEGreedy
from policies.Softmax import Softmax
from policies.TimeSoftmax import TimeSoftmax
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
import sys
import os

grid = []

with open("grid.txt") as f:
    for line in f:
        grid.insert(0,literal_eval(line[:-1])) #Opposite order
grid_size = np.shape(grid)

#printer monitor
printer = Printer()
printer.follow(episode_ended)
printer.activate()

flags = []
for row in grid:
    for pos in row:
        if pos[1]:
            flags.append(pos[1])

pos_length = len(bin(grid_size[0]*grid_size[1]-1)[2:])
num_flags = len(flags)

def state_to_strips(state):
    agent_idx = np.where(state == 1)[0][0]
    flag_idx = agent_idx / (grid_size[0]*grid_size[1])
    pos_idx = agent_idx % (grid_size[0]*grid_size[1])
    pos = [pos_idx % grid_size[0], pos_idx / grid_size[0]]
    # convert flag_idx in flags taken
    taken = bin(flag_idx)[2:]
    strips_state = set([])
    shift = num_flags - len(taken)
    for i in range(len(taken)):
        if taken[i] == '1':
            strips_state.add('taken_flag' + flags[i+shift])
    strips_state.add('in_' + grid[pos[0]][pos[1]][2])
    return strips_state

actions = [0, 1, 2, 3]

state_length = grid_size[0]*grid_size[1]*(2**num_flags)

experiments = {}

TimeEGreedy_formula = lambda t:1./np.sqrt(t)
TimeSoftmax_formula = lambda t:4*((1000-float(t))/1000) + 1

#### Experiment 1: 2 agents with no plan or heuristic
#####################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    learners.append(Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

no_plan = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 1}
experiments['no_plan'] = no_plan

#### Experiment 2: 2 agents with joint-plan
###########################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan'] = joint_plan

#### Experiment 3: 2 agents with individual-plans
#################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan'] = individual_plan

#### Experiment 4: 2 agents with joint-plan and flag-heuristic
##############################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_flags = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_flags'] = joint_plan_flags

#### Experiment 5: 2 agents with individual-plans and flag-heuristic
####################################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_flags = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_flags'] = individual_plan_flags

#### Experiment 6: 2 agents with flag-heuristic
###############################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    shaper = FlagShaper(num_flags=num_flags)
    e_greedy = EGreedy(epsilon=0.1)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

flag_based = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['flag_based'] = flag_based

#### Experiment 7: 2 agents with no plan or heuristic with Time E-Greedy
#####################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    learners.append(Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

no_plan_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 1}
experiments['no_plan_time_egreedy'] = no_plan_time_egreedy

#### Experiment 8: 2 agents with joint-plan with Time E-Greedy
###########################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_time_egreedy'] = joint_plan_time_egreedy

#### Experiment 9: 2 agents with individual-plans with Time E-Greedy
#################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_time_egreedy'] = individual_plan_time_egreedy

#### Experiment 10: 2 agents with joint-plan and flag-heuristic with Time E-Greedy
##############################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_flags_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_flags_time_egreedy'] = joint_plan_flags_time_egreedy

#### Experiment 11: 2 agents with individual-plans and flag-heuristic with Time E-Greedy
####################################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_flags_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_flags_time_egreedy'] = individual_plan_flags_time_egreedy

#### Experiment 12: 2 agents with flag-heuristic with Time E-Greedy
###############################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    shaper = FlagShaper(num_flags=num_flags)
    e_greedy = TimeEGreedy(formula=TimeEGreedy_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=e_greedy, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

flag_based_time_egreedy = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['flag_based_time_egreedy'] = flag_based_time_egreedy


#### Experiment 13: 2 agents with no plan or heuristic, using Softmax policy
#####################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    learners.append(Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

no_plan_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 1}
experiments['no_plan_softmax'] = no_plan_softmax

#### Experiment 14: 2 agents with joint-plan, using Softmax policy
###########################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_softmax'] = joint_plan_softmax

#### Experiment 15: 2 agents with individual-plans, using Softmax policy
#################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_softmax'] = individual_plan_softmax

#### Experiment 16: 2 agents with joint-plan and flag-heuristic, using Softmax policy
##############################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_flags_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_flags_softmax'] = joint_plan_flags_softmax

#### Experiment 17: 2 agents with individual-plans and flag-heuristic, using Softmax policy
####################################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_flags_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_flags_softmax'] = individual_plan_flags_softmax

#### Experiment 18: 2 agents with flag-heuristic, using Softmax policy
###############################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    shaper = FlagShaper(num_flags=num_flags)
    softmax = Softmax(temperature=1.0)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

flag_based_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['flag_based_softmax'] = flag_based_softmax

#### Experiment 19: 2 agents with no plan or heuristic with Time Softmax
#####################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    learners.append(Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

no_plan_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 1}
experiments['no_plan_time_softmax'] = no_plan_time_softmax

#### Experiment 20: 2 agents with joint-plan with Time Softmax
###########################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length=state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_time_softmax'] = joint_plan_time_softmax

#### Experiment 21: 2 agents with individual-plans with Time Softmax
#################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = STRIPS(strips_plan=strips_plan, convert=state_to_strips, omega=600./len(strips_plan))
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_time_softmax'] = individual_plan_time_softmax

#### Experiment 22: 2 agents with joint-plan and flag-heuristic with Time Softmax
##############################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_joint[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

joint_plan_flags_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['joint_plan_flags_time_softmax'] = joint_plan_flags_time_softmax

#### Experiment 23: 2 agents with individual-plans and flag-heuristic with Time Softmax
####################################################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    strips_plan = strips_plans_individual[i]
    shaper = CombinedShaper(strips_plan=strips_plan, convert=state_to_strips, num_flags=num_flags,
                            omega=600./(float (len(strips_plan) + num_flags)))
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins,goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

individual_plan_flags_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['individual_plan_flags_time_softmax'] = individual_plan_flags_time_softmax

#### Experiment 24: 2 agents with flag-heuristic with Time Softmax
###############################################

nr_of_agents = 2
learners = []
begins = [[7,4], [7,14]]

for i in range(nr_of_agents):
    shaper = FlagShaper(num_flags=num_flags)
    softmax = TimeSoftmax(formula=TimeSoftmax_formula)
    no_features = Feature(state_length= state_length)
    trace = Eligibility(lambda_=0.4, actions=actions, shape=(no_features.num_features(), len(actions)))
    sarsa = Sarsa(actions=actions, alpha=0.1, gamma=0.99, policy=softmax, features=no_features, trace=trace)
    learners.append(RewardShaping(learner=sarsa, shaper=shaper))

environment = RoomWorld(grid, population_size=nr_of_agents, ids=[l.id for l in learners], begins=begins, goal=[1,1])
experiment = MultiGeneric(max_steps=None, episodes=1000, trials=30, learners=learners, environment=environment)

flag_based_time_softmax = {'experiment': experiment, 'nr_of_agents': nr_of_agents, 'learners_per_agent': 2}
experiments['flag_based_time_softmax'] = flag_based_time_softmax

#### Run Experiment
###################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Please provide experiment names. Choose out of the following:\n" + '\n'.join(experiments.keys())
    else:
        for experiment_name in sys.argv[1:]:
            writer = Writer(folderpath=experiment_name)
            writer.follow(episode_ended)
            writer.activate()
            print "Executing experiment: " + experiment_name
            experiment = experiments[experiment_name]['experiment']
            experiment.run()
            if not os.path.exists(experiment_name): #Make directory with experiment_name if necessary
                os.makedirs(experiment_name)
            writer.save()