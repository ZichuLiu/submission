#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -a 0-53
#SBATCH -p rtx6000
#SBATCH --qos normal
#SBATCH -o PPM_Adam-%A_%a-result.out
#SBATCH --error PPM_Adam-%A_%a-result.out
#SBATCH --open-mode=append
export LD_LIBRARY_PATH=/pkgs/cuda-10.2/lib64:/pkgs/cudnn-10.2-v7.6.5/lib64:$LD_LIBRARY_PATH
export PATH=/pkgs/cuda-10.2/bin:$PATH
export LIBRARY_PATH=/pkgs/cuda-10.2/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}

BETA1=('0.0' '0.1' '0.2' '0.3' '0.4' '0.5')
BETA2=('0.9' '0.95' '0.99')
NUM_STEP=('3' '4' '5')
export PREEMPT=True
for i in {0..5}
do
  for j in {0..2}
  do
    for k in {0..2}
      do
            let task_id=$k+3*$j+9*$i
            if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
              echo $task_id
              source activate sinkhorn

              beta1=${BETA1[$i]}
              beta2=${BETA2[$j]}
              num_step=${NUM_STEP[$k]}
              expname=cifar_ppm_Adam_beta1"$beta1"_beta2"$beta2"_num_step_"$num_step"
              python -u main.py  --exp_name $expname --beta1 $beta1 --beta2 $beta2 --alg ppm --load_path $expname --num_workers 0 --val_freq 1 --extra_steps $num_step
            fi

      done
  done
done