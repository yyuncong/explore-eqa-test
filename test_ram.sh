#!/bin/bash
#SBATCH --job-name=ram_test
#SBATCH -o ram_output/ram_test_%j.out
#SBATCH -e ram_output/ram_test_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
conda activate explore-eqa
module load gcc/13.2.0

python run_data_collector_clustering.py -cf cfg/data_collection_pathfinder_clustering.yaml