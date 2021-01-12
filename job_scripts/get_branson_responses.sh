#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=get_branson_responses
#SBATCH --partition=trc
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/mhturner/SC-FC/job_outputs/%x.%j.out
#SBATCH --open-mode=append

ml python/3.6.1

python3 /home/users/mhturner/SC-FC/get_branson_responses.py
