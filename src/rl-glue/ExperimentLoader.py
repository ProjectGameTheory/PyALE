from RLGlueExperiment import RLGlueExperiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run RL-Glue Experiment')
    parser.add_argument('--max_steps', metavar='S', type=int,
                        default=None, help='maximum steps per trial')
    parser.add_argument('--trials', metavar='T', type=int,
                        default=1, help='number of independent trials')
    parser.add_argument('--episodes', metavar='E', type=int,
                        default=100, help='number of episodes per trial')
    args = parser.parse_args()

    experiment = RLGlueExperiment(
        max_steps=args.max_steps, trials=args.trials, episodes=args.episodes)
    experiment.run()