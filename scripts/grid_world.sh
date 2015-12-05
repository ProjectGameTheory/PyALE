#!/bin/sh
# Submit the job to all.q
#$ -clear
#$ -q all.q
# use the current working directory for input and output
#$ -cwd
# export the current $PATH variable for this job
#$ -V
#$ -S /bin/bash
# Set a custom job name
#$ -N grid-world
cd ../src
AGENT="rl-glue/AgentLoader.py"
AGENT_OPTIONS='--type Q --alpha 0.1 --gamma 0.9 --lambda_ 0.0 
--policy EGreedy --epsilon 0.2 
--actions 0 1 2 3 4 5 6 7 --observation 0 84'
ENVIRONMENT="rl-glue/EnvironmentLoader.py"
ENVIRONMENT_OPTIONS="--type WindyWorld --size 12 7 --start 0 3 --goal 9 3 
--wind 0 0 1 1 1 2 2 1 0 0 0 0"
EXPERIMENT="rl-glue/ExperimentLoader.py"
EXPERIMENT_OPTIONS="--max_steps 0  --episodes 5000 --trials 1"
#export PYTHONPATH=~/rl-glue-2.02/src:$PYTHONPATH

if [ -n "$1" ]
then
    export RLGLUE_PORT=$1
else
    export RLGLUE_PORT=2048
fi

rl_glue & 
python $AGENT $AGENT_OPTIONS & 
python $ENVIRONMENT $ENVIRONMENT_OPTIONS & 
python $EXPERIMENT $EXPERIMENT_OPTIONS