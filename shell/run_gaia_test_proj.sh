#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o gaia_test_proj_snc_output_3000.txt
#SBATCH --job-name=snc_proj



cd notebooks/gaia_test
python gaia_test_extra_proj_snc.py


