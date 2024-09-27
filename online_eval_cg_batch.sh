l=$1

for (( i=0; i<l; i++ )); do
  # Submit the job using sbatch
  job_id=$(sbatch online_eval_cg.sh $i $l| awk '{print $4}')
  echo "Submitting job for tasks[$i] with $job_id"

done