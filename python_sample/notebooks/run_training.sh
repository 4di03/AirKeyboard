#!/bin/bash
#SBATCH -J Train_model              # Job name
#SBATCH -N 1              # Number of nodes
#SBATCH -n 1                 # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH -p gpu 
#SBATCH --time=07:59:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -o /scratch/palle.a/AirKeyboard/logs/py_train_out_%j.txt       # Standard output file
#SBATCH -e /scratch/palle.a/AirKeyboard/logs/py_train_error_%j.txt        # Standard error file


module load anaconda3
source activate cv
# ARG 1 is path to config file for trianing (find in python_samples/notebooks/samples)
python train.py $1 