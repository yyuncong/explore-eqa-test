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

# python online_evaluation_ours_new_1.py -cf cfg/online_eval_ours_new.yaml
# python online_evaluation_ours_3.py -cf cfg/online_eval_ours_2.yaml
# python online_evaluation_openeqa_3.py -cf cfg/online_eval_openeqa.yaml
# python online_evaluation_ours_dynamic.py -cf cfg/online_eval_ours_dynamic.yaml
# python online_evaluation_openeqa_new.py -cf cfg/online_eval_openeqa_new.yaml
python online_evaluation_openeqa_clustering.py -cf cfg/online_eval_openeqa_clustering.yaml
# python online_evaluation_openeqa_dynamic.py -cf cfg/online_eval_openeqa_dynamic.yaml
# python online_evaluation_openeqa_new_logic.py -cf cfg/online_eval_openeqa_new_logic.yaml
# python run_openeqa_clustering.py -cf cfg/openeqa_clustering.yaml
# python online_evaluation_ours_clustering.py -cf cfg/online_eval_ours_clustering.yaml