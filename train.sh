#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=01-23:59:59
#SBATCH --job-name=SD_decoder_zeroConv_Freeze
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

# training
python3 -u train.py \
  --batch_size 2 \
  --lr 1e-5 \
  --logger_freq 250 \
  --sd_locked true \
  --only_mid_control false \
  --gpu 2 \
  --min_epoch 1 \
  --max_epoch 8 \
  --exp_name "DECODER WITH ZERO CONV, FREEZE" \