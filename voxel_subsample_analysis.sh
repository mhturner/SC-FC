#!/bin/bash
#
#SBATCH --job-name=vox_subsample
#SBATCH --partition=trc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/users/mhturner/SC-FC/job_outputs/%x.%j.out
#SBATCH --open-mode=append

ml python/3.6.1

python3 /home/users/mhturner/SC-FC/voxel_subsample_analysis.py
