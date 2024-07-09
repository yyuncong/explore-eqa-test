import os
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--num_split",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-c",
        "--command",
        type=str,
        default="python run_data_collector_new.py -cf cfg/data_collection_pathfinder_new_aimos.yaml --path_id_offset 0 --seed 0"
    )

    parser.add_argument(
        "-j",
        "--job_name",
        type=str,
        required=False,
        default='gen_path'
    )

    args = parser.parse_args()

    num_split = args.num_split
    command = args.command
    job_name = args.job_name

    print(f'Running command [{command}] for {num_split} splits')

    increment = 1 / num_split
    start_ratio = 0.0
    job_count = 1
    while start_ratio < 1.0 - 1e-3:
        end_ratio = start_ratio + increment

        with open('run_test_jobs.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'#SBATCH --job-name={job_name}_{job_count}\n' +
                    f'#SBATCH -o output/{job_name}_{job_count}_%j.out\n' +
                    f'#SBATCH -e output/{job_name}_{job_count}_%j.err\n' +
                    '#SBATCH --mem=100G\n' +
                    '#SBATCH --nodes=1\n' +
                    '#SBATCH --ntasks-per-node=1\n' +
                    '#SBATCH --time=06:00:00\n' +
                    '#SBATCH --gres=gpu:1\n' +
                    '#SBATCH --cpus-per-task=4\n\n')

            f.write(f'{command} ' + '--start %.3f --end %.3f\n\n' % (start_ratio, end_ratio))

        os.system('cat run_test_jobs.sh')
        os.system('sbatch run_test_jobs.sh')
        time.sleep(10)
        start_ratio += increment
        job_count += 1

    os.system('rm run_test_jobs.sh')