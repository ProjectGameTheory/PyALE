from learners.Sarsa import Sarsa
from learners.Q import Q
from learners.ExperienceReplay import ExperienceReplay
from learners.RewardShaping import RewardShaping
from policies.EGreedy import EGreedy
from features.Feature import Feature
from traces.Eligibility import Eligibility
from rlglue.agent import AgentLoader as AgentLoader
from RLGlueAgent import RLGlueAgent
import argparse

from monitors.Event import episode_ended
from monitors.Printer import Printer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run an RL-Glue Agent')
    # Sarsa, Q, Sarsa ER, Q ER, Sarsa RS, Q RS
    parser.add_argument('--type', metavar='T', type=str,
                        default=['Sarsa'], nargs='*', help='type of agent')
    parser.add_argument('--gamma', metavar='G', type=float,
                        default=0.999, help='discount factor')
    parser.add_argument('--alpha', metavar='A', type=float,
                        default=0.5, help='learning rate')
    parser.add_argument('--lambda_', metavar='L', type=float,
                        default=0.9, help='trace decay')
    parser.add_argument('--epsilon', metavar='E', type=float,
                        default=0.05, help='exploration rate')
    # None, TileCoding, RBF, RAM, Basic, BASS
    parser.add_argument('--features', metavar='F', type=str,
                        default='None', help='features to use')
    parser.add_argument('--actions', metavar='C', type=int,
                        default=None, nargs='*', help='list of allowed actions')
    # Random, EGreedy
    parser.add_argument('--policy', metavar='P', type=str,
                        default='EGreedy', help='agent policy')
    # Eligibility
    parser.add_argument('--trace', metavar='TR', type=str,
                        default='Eligibility', help='type of trace used')
    # 33728 is size of ALE observation
    parser.add_argument('--observation', metavar='O', type=int,
                        default=[0, 33728], nargs='*', help='size of an observation')
    parser.add_argument('--replays', metavar='RE', type=int,
                        default=1, help='number of replays for ER')
    parser.add_argument('--max_samples', metavar='MS', type=int,
                        default=None, help='max size of sample db for ER')
    # TODO add shapers
    parser.add_argument('--shaper', metavar='SH', type=str,
                        default=None, help='the shaper used for RS')
    parser.add_argument('--omega', metavar='OM', type=float,
                        default=1.0, help='scaling factor')
    args = parser.parse_args()

    # monitors
    printer = Printer()
    printer.follow(episode_ended)
    printer.activate()

    observation_size = args.observation[1] - args.observation[0]

    policies = {'EGreedy': EGreedy}
    p_kwargs = {'EGreedy': {'epsilon': args.epsilon}}
    policy = policies[args.policy](**p_kwargs[args.policy])

    features = {'None': Feature}
    f_kwargs = {'None': {'state_length': observation_size}}
    feature = features[args.features](**f_kwargs[args.features])

    traces = {'Eligibility': Eligibility}
    t_kwargs = {'Eligibility': {'lambda_': args.lambda_,
                                'actions': args.actions,
                                'shape': (feature.num_features(), len(args.actions))}}
    trace = traces[args.trace](**t_kwargs[args.trace])

    agents = {'Sarsa': Sarsa, 'Q': Q,
              'ER': ExperienceReplay, 'RS': RewardShaping}
    q_sarsa_kwargs = {'actions': args.actions,
                      'alpha': args.alpha,
                      'gamma': args.gamma,
                      'policy': policy,
                      'trace': trace,
                      'features': feature}
    a_kwargs = {'Sarsa': q_sarsa_kwargs,
                'Q': q_sarsa_kwargs,
                'ER': {'replays': args.replays, 'max_samples': args.max_samples}}
    agent = agents[args.type[0]](**a_kwargs[args.type[0]])
    if len(args.type) == 2:
        agent = agents[args.type[1]](learner=agent, **a_kwargs[args.type[1]])
    AgentLoader.loadAgent(RLGlueAgent(
        learner=agent, observation_limits=args.observation))
