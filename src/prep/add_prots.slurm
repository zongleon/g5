#!/bin/bash
#SBATCH -J annot -p standard
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G
#SBATCH -o annot_output --mail-type=end --mail-user=lzong@u.rochester.edu

python3 add_prots.py
