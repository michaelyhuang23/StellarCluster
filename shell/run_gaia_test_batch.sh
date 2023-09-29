#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o gaia_test_batch_snc_output.txt
#SBATCH --job-name=snc

pwd

module load anaconda/2023a


python notebooks/gaia_test/gaia_test_batch_snc.py

