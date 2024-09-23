#!/bin/bash
#SBATCH -J g5tknds -p standard
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G
#SBATCH -o tkns_ds_output --mail-type=end --mail-user=lzong@u.rochester.edu

python3 tokenize_dataset.py
