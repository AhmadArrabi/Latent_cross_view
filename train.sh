#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01-23:59:59
#SBATCH --job-name=SD_zeroConv_unFreeze
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

# training
python3 -u train.py \
  --batch_size 2 \
  --lr 1e-5 \
  --logger_freq 500 \
  --sd_locked false \
  --only_mid_control false \
  --gpu 4 \
  --min_epoch 1 \
  --max_epoch 8