#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=01-23:59:59
#SBATCH --job-name=GeoMAP
#SBATCH --mail-user=aarrabi@uvm.edu
#SBATCH --mail-type=ALL

# training
python3 -u train.py \
  --batch_size 2 \
  --lr 1e-5 \
  --logger_freq 200 \
  --gpu 2 \
  --min_epoch 1 \
  --max_epoch 8 \
  --exp_name "RESNET GEOMAPPING" \