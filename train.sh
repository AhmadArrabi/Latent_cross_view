#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=01-23:59:59
#SBATCH --job-name=Stable_cross_view
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

# training
python3 -u train.py \
  --batch_size 4 \
  --lr 1e-5 \
  --logger_freq 400 \
  --sd_locked true \
  --only_mid_control false \
  --gpu 4 \
  --min_epoch 1 \
  --max_epoch 8