#!/bin/bash
#SBATCH --job-name=zjc_ego_mem
#SBATCH -o output/run_%j.out
#SBATCH -e output/run_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
conda activate explore-eqa

python online_evaluation_openeqa_3.py -cf cfg/online_eval_openeqa.yaml
