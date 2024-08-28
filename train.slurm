#!/bin/bash
#SBATCH -J g5
#SBATCH -t 08:00:00
#SBATCH -p h100 --gres=gpu:4
#SBATCH --mem=32G
#SBATCH -o train_%j --mail-type=end --mail-user=lzong@u.rochester.edu

torchrun --nproc_per_node 4 train.py
