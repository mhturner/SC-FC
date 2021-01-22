#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=hemi_jfrc2
#SBATCH --partition=trc
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/users/mhturner/SC-FC/job_outputs/%x.%j.out
#SBATCH --open-mode=append

module use /home/groups/trc/modules
ml R/3.6.1
module load cmtk
module load hdf5/1.10.6
module load fftw/3.3.8

Rscript /home/users/mhturner/SC-FC/hemibrain_2_atlas.r JFRC2
