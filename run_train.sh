#!/bin/bash
#SBATCH -J Train_model              # Job name
#SBATCH -N 1              # Number of nodes
#SBATCH -n 1                 # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH -p gpu 
#SBATCH --mem=50G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -o /scratch/palle.a/AirKeyboard/logs/train_out_%j.txt       # Standard output file
#SBATCH -e /scratch/palle.a/AirKeyboard/logs/train_error_%j.txt        # Standard error file
source setup_env.sh
cd build
bash build_and_run.sh $1 $2 $3
