#!/bin/sh
JOB_NAME="alejob"
if [ -n "$1" ]
then
    JOB_NAME=$1
fi
# Submit the job to all.q
#$ -clear
#$ -q all.q
# use the current working directory for input and output
#$ -cwd
# export the current $PATH variable for this job
#$ -V
#$ -S /bin/bash
# Set a custom job name
#$ -N $JOB_NAME
#$ -l virtual_free=1G
# stdout and stderr output
#$ -e $HOME/Logs/$JOB_NAME.err
#$ -o $HOME/Logs/$JOB_NAME.log
#separate threads for processes
#$ -pe mt 4
#
#set various paths
RLDIR=$HOME/mals/assignment2/PyALE/src
ALEDIR=$HOME/ale/ale_0_5_1
LOGDIR=$HOME/logs

#Experimental configuration
EXP_NAME="ersarsa"
EXPERIMENT="generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 18000  --numeps 3000 --numtrials 5"
AGENT="agents/ALEERSarsaAgent.py"
AGENT_OPTIONS='--eps 0.05 --lambda_ 0.5 --alpha 0.1 --actions 0 1 3 4'
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0"
GAME="space_invaders.bin"

#######Imports#################
#source $HOME/.bash_profile
#module add java
export PYTHONPATH=$RLDIR:$PYTHONPATH
#export PYTHONPATH=~/rl-glue-2.02/src:$PYTHONPATH

if [ -n "$2" ]
then
    export RLGLUE_PORT=$2
else
    export RLGLUE_PORT=1028
fi
###############################

mkdir -p "$LOGDIR/$EXP_NAME"

cd $RLDIR
#start rlglue
rl_glue&
#run agent
python $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/$EXP_NAME" > $LOGDIR/agent-$EXP_NAME.log 2>&1 &
#run experiment
cd exp
python $EXPERIMENT $EXPERIMENT_OPTIONS  >> $LOGDIR/exp-$EXP_NAME.log 2>&1 &
#run environment (no '&' or job quits!)
cd $ALEDIR
./ale $ALE_OPTIONS roms/$GAME > $LOGDIR/ale-$EXP_NAME.log 2>&1
