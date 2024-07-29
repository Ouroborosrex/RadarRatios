#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script for Arc nodes
#SBATCH -J CommsRadarGen # Job Name
#SBATCH -o dfd_job.o%j # Name of the stdout output file
#SBATCH -e dfd_job.e%j # Name of stderr error file
#SBATCH -p compute1 # Queue (partition) name
#SBATCH -N 1 # Total # of nodes (must be 1 for serial)
#SBATCH -n 40 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 28:00:00 # Run time (hh:mm:ss)
module load anaconda3
conda init
conda activate myvm
echo Creating 2D and 1D datasets...
# Launch code...
python3 dataset_generation.py 

