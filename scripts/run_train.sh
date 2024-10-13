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

source setup_env.sh
cd build
# cmake ..
bash build_and_run.sh $1 $2 $3 $4 $5 $6 $7

# Sample Usage:

# SMALL_DATA RUN:
# bash run_train.sh iou default_model.pt --no-reload 


# SUCCESSFUL RUN (0.60 loss):

"""
model builder initalization:

CuNetBuilder* modelBuilder = new CuNetBuilder();
modelBuilder->inChannels = channels;
modelBuilder->outChannels = 21;
modelBuilder->initNeurons = 64;

# 4 levels by default
"""

# bash run_train.sh iou default_model.pt --no-reload /scratch/palle.a/PalmPilot_Data/data_tensors/mid_data