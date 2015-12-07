from learners.Q import Q
from learners.ExperienceReplay import ExperienceReplay
from policies.EGreedy import EGreedy
from traces.Eligibility import Eligibility
from features.Feature import Feature
from environments.WindyWorld import WindyWorld
from experiments.Generic import Generic

from monitors.Event import episode_ended
from monitors.Printer import Printer
from monitors.Writer import Writer

actions = [0, 1, 2, 3, 4, 5, 6, 7]
grid = [12, 7]

#monitors
printer = Printer()
printer.follow(episode_ended)
printer.activate()
writer = Writer()
writer.follow(episode_ended)
writer.activate()

#agent
e_greedy = EGreedy(epsilon=0.2)
no_features = Feature(state_length= grid[0]*grid[1])
trace = Eligibility(lambda_=0.0, actions=actions, shape=(no_features.num_features(), len(actions)))
q = Q(actions=actions, alpha=0.1, gamma=0.9, policy=e_greedy, features=no_features, trace=trace)
learner = ExperienceReplay(learner=q, replays=2, max_samples=None)

environment = WindyWorld(size=grid, begin=[0,3], goal=[9,3], wind=[0, 0, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0])

experiment = Generic(max_steps=None, episodes=5000, trials=1, learner=q, environment=environment)

#run generic experiment
experiment.run()
writer.save()