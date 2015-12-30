import numpy as np
import matplotlib.pyplot as pl
from strips import experiments
import pandas
import os
from scipy.stats import sem


pl.figure()
ax = pl.subplot(111)
for experiment_name, experiment_properties in experiments.iteritems():
    if not os.path.isdir(experiment_name):
        continue
    print experiment_name
    df = pandas.read_csv(experiment_name + '/episode.csv', index_col=0)[::experiment_properties['learners_per_agent']]
    df.set_index([range(1,len(df)+1)], inplace=True)

    a = df.index.values
    idx = np.array([a] * experiment_properties['nr_of_agents']).T.flatten()[:len(a)]
    df = df.groupby(idx).mean()
    trial_means = pandas.rolling_mean(df.groupby('episode').reward.mean().values, 100)
    #trial_medians = pandas.rolling_mean(df.groupby('episode').reward.median().values, 5)
    trial_sems = pandas.rolling_mean(df.groupby('episode').reward.apply(sem).mul(1.96).values, 100)
    episodes = df.groupby('episode').reward.mean().keys()
    # pl.fill_between(episodes, trial_means - trial_sems,
    #              trial_means + trial_sems, color="#3F5D7D")
    pl.plot(episodes, trial_means, lw=1,label=experiment_name)

#pl.ylim(0,700)
pl.xlabel('Episode')
pl.ylabel('Discounted total reward per episode')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
pl.show()


pl.figure()
ax = pl.subplot(111)
for experiment_name, experiment_properties in experiments.iteritems():
    if not os.path.isdir(experiment_name):
        continue
    print experiment_name
    df = pandas.read_csv(experiment_name + '/episode.csv', index_col=0)[::experiment_properties['learners_per_agent']]
    df.set_index([range(1,len(df)+1)], inplace=True)

    a = df.index.values
    idx = np.array([a] * experiment_properties['nr_of_agents']).T.flatten()[:len(a)]
    df = df.groupby(idx).mean()
    trial_means = pandas.rolling_mean(df.groupby('episode').steps.mean().values, 20)
    #trial_medians = pandas.rolling_mean(df.groupby('episode').steps.median().values, 5)
    trial_sems = pandas.rolling_mean(df.groupby('episode').steps.apply(sem).mul(1.96).values, 20)
    episodes = df.groupby('episode').steps.mean().keys()
    # pl.fill_between(episodes, trial_means - trial_sems,
    #              trial_means + trial_sems, color="#3F5D7D")
    pl.plot(episodes, trial_means, lw=1,label=experiment_name)

pl.ylim(0,1000)
pl.xlabel('Episode')
pl.ylabel('Discounted steps per episode')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
pl.show()