#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script for Arc nodes
#SBATCH -J train_wave # Job Name
#SBATCH -o train_job.o%j # Name of the stdout output file
#SBATCH -e train_job.e%j # Name of stderr error file
#SBATCH -p gpu1v100 # Queue (partition) name
#SBATCH -N 1 # Total # of nodes (must be 1 for serial)
#SBATCH -n 40 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00 # Run time (hh:mm:ss)
module load anaconda3
conda init
conda activate myvm
# Launch code...
python3 train_all_models.py
