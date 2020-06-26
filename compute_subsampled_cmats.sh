#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=cmats_subsampled
#SBATCH --partition=trc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/users/mhturner/SC-FC/job_outputs/%x.%j.out
#SBATCH --open-mode=append

ml python/3.6.1

python3 /home/users/mhturner/SC-FC/compute_subsampled_cmats.py
