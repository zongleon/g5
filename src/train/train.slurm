#!/bin/bash
#SBATCH -J g5
#SBATCH -t 2-00:00:00
#SBATCH -p h100 --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -o train_%j --mail-type=end --mail-user=lzong@u.rochester.edu

torchrun --nproc_per_node 4 train.py
