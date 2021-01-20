#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=r_hemi
#SBATCH --partition=trc
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/users/mhturner/SC-FC/job_outputs/%x.%j.out
#SBATCH --open-mode=append

ml R

Rscript /home/users/mhturner/SC-FC/hemibrain_2_jfrc2.r
