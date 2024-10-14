#!/bin/bash
#SBATCH -J Train_model              # Job name
#SBATCH -N 1              # Number of nodes
#SBATCH -n 1                 # Number of tasks
#SBATCH --cpus-per-task=8
#SBATCH -p gpu 
#SBATCH --time=07:59:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -o /scratch/palle.a/PalmPilot/logs/train_out_%j.txt       # Standard output file
#SBATCH -e /scratch/palle.a/PalmPilot/logs/train_error_%j.txt        # Standard error file

source scripts/setup_env.sh
# see sample input file structure from run_train.yaml
INPUT_FILE=$(realpath $1)

# must be set to --debug for gdb mode to run
DEBUG=$2

# cmake ..
bash scripts/build_and_run.sh $INPUT_FILE $DEBUG

# Sample Usage:

# SMALL_DATA RUN:
# bash scripts/run_train.sh scripts/inputs/train_default.json --debug


# SUCCESSFUL RUN (0.60 loss):


# bash scripts/run_train.sh iou default_model --no-reload /scratch/palle.a/PalmPilot_Data/data_tensors/mid_data

# 44734450 is full run with these params:
# sbatch scripts/run_train.sh iou ft_model --no-reload /scratch/palle.a/PalmPilot_Data/data_tensors/full_data