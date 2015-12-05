from environments.GridWorld import GridWorld
from environments.WindyWorld import WindyWorld
from RLGlueEnvironment import RLGlueEnvironment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run an RL-Glue Environment')
    # GridWorld, WindyWorld
    parser.add_argument('--type', metavar='T', type=str, default='GridWorld')
    parser.add_argument('--size', metavar='Z', type=int,
                        default=[10, 10], nargs='*')
    parser.add_argument('--start', metavar='S', type=int,
                        default=[0, 0], nargs='*')
    parser.add_argument('--goal', metavar='G', type=int,
                        default=[9, 9], nargs='*')
    parser.add_argument('--wind', metavar='W', type=int,
                        default=None, nargs='*')
    args = parser.parse_args()

    envs = {'GridWorld': GridWorld,
            'WindyWorld': WindyWorld}
    kwargs = {'GridWorld': {'size': args.size, 'begin': args.start, 'goal': args.goal},
              'WindyWorld': {'size': args.size, 'begin': args.start, 'goal': args.goal, 'wind': args.wind}}
    env = envs[args.type](**kwargs[args.type])
    EnvironmentLoader.loadEnvironment(RLGlueEnvironment(environment=env))
