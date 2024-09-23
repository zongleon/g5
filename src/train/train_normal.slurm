#!/bin/bash
#SBATCH -J g5
#SBATCH -t 2-00:00:00
#SBATCH -p gpu --gres=gpu:2 -C V100|A100
#SBATCH --mem=64G
#SBATCH -o train_output%j --mail-type=end --mail-user=lzong@u.rochester.edu

torchrun --nproc_per_node 2 train_normal.py
