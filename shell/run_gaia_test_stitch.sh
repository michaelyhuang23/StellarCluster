#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o gaia_test_stitch_snc_output_random_sparsification_3000.txt
#SBATCH --job-name=snc_stitch



cd notebooks/gaia_test
python gaia_test_stitch_snc.py

