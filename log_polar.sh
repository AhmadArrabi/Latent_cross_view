#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01-23:59:59
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --job-name=layout
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

python3 -u layout_model.py