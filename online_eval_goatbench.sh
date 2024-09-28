#!/bin/bash
#SBATCH --job-name=eval_goatbench
#SBATCH -o output/goatbench_%j.out
#SBATCH -e output/goatbench_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
module load gcc/13.2.0
conda activate explore-eqa

split_index=$1
total_splits=$2

python online_evaluation_goatbench.py \
-cf cfg/online_eval_goatbench.yaml \
--start_ratio 0.0 \
--end_ratio 0.1 \
--split_index $split_index \
--split_number $total_splits
