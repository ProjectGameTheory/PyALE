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
#$ -N er-si
RLDIR=$HOME/mals/assignment2/PyALE/src
ALEDIR=$HOME/ale/ale_0_5_1

AGENT="rl-glue/AgentLoader.py"
#AGENT='agents/ALESarsaAgent.py'
#AGENT_OPTIONS='--alpha 0.1 --gamma 0.999 --lambda_ 0.5 --eps 0.05 --features RAM --actions 0 1 3 4'
AGENT_OPTIONS='--type Sarsa --alpha 0.1 --gamma 0.999 --lambda_ 0.5 --normalization True
--replays 2 --max_samples 5000
--policy EGreedy --epsilon 0.05 --features RAM
--actions 0 1 3 4'
EXPERIMENT="rl-glue/ExperimentLoader.py"
EXPERIMENT_OPTIONS="--max_steps 18000  --episodes 3000 --trials 5"
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0"
GAME="space_invaders.bin"
#export PYTHONPATH=~/rl-glue-2.02/src:$PYTHONPATH

#######Imports#################
#source $HOME/.bash_profile
#module add java
export PYTHONPATH=$RLDIR:$PYTHONPATH

if [ -n "$1" ]
then
    export RLGLUE_PORT=$1
else
    export RLGLUE_PORT=2048
fi

cd $RLDIR
#start rlglue
rl_glue & 
python $AGENT $AGENT_OPTIONS & 
python $EXPERIMENT $EXPERIMENT_OPTIONS & 
cd $ALEDIR
./ale $ALE_OPTIONS roms/$GAME