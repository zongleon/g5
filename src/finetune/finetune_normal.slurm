#!/bin/bash
#SBATCH -J g5
#SBATCH -t 3-00:00:00
#SBATCH -p gpu --gres=gpu:2 -C V100|A100
#SBATCH --mem=32G
#SBATCH -o finetune_%j --mail-type=end --mail-user=lzong@u.rochester.edu

torchrun --nproc_per_node 2 finetune.py
