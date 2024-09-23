#!/bin/bash
#SBATCH -J g5tkns
#SBATCH -t 02:00:00
#SBATCH --mem=128G
#SBATCH -o tkns_output --mail-type=end --mail-user=lzong@u.rochester.edu

python3 tokenizer.py
