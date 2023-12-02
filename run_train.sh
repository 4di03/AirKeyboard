#!/bin/bash
#SBATCH -J Train_model              # Job name
#SBATCH -N 1              # Number of nodes
#SBATCH -n 1                 # Number of tasks
#SBATCH -p gpu 
#SBATCH --gres=gpu:p100:1
#SBATCH -o /scratch/palle.a/AirKeyboard/logs/train_out_%j.txt       # Standard output file
#SBATCH -e /scratch/palle.a/AirKeyboard/logs/train_error_%j.txt        # Standard error file
source setup_env.sh
build/Open_CV_Project
