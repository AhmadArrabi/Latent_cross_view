#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01-23:59:59
#SBATCH --mem=64G
#SBATCH --job-name=log_polar_seattle
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

python3 transformation.py