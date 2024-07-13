#!/bin/bash
#SBATCH --job-name=zjc_ego_mem
#SBATCH -o output/run_%j.out
#SBATCH -e output/run_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="a100"

module load miniconda/22.11.1-1
conda activate explore-eqa

python online_evaluation_ours_2.py -cf cfg/online_eval_ours_2.yaml
