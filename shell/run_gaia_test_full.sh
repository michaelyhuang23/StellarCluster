#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o gaia_test_full_snc_output.txt
#SBATCH --job-name=snc

pwd

module load anaconda/2023a

cd notebooks/gaia_test
python gaia_test_full_snc.py

