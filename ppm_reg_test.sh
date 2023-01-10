#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -a 0-1
#SBATCH -p rtx6000
#SBATCH --qos normal
#SBATCH -o PPM_Adam_beta1ema-%A_%a-result.out
#SBATCH --error PPM_Adam_beta1ema-%A_%a-result.out
#SBATCH --open-mode=append
export LD_LIBRARY_PATH=/pkgs/cuda-10.2/lib64:/pkgs/cudnn-10.2-v7.6.5/lib64:$LD_LIBRARY_PATH
export PATH=/pkgs/cuda-10.2/bin:$PATH
export LIBRARY_PATH=/pkgs/cuda-10.2/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}


#BETA2=('0.9' '0.95' '0.98' '0.99') de 0.9
#NUM_STEP=('4' '5' '6') #de 5
EMA=('0.999' '0.9999')
#
export PREEMPT=True


for k in {0..1}
  do
        let task_id=$k
        if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
        then
          echo $task_id
          source activate sinkhorn

          beta1='-0.1'
          beta2='0.9'
          num_step='6'
          ema=${EMA[$j]}
          expname=cifar_ppm_Adam_beta1"$beta1"_beta2"$beta2"_num_step_"$num_step"_ema_"$ema"
          python -u main.py  --exp_name $expname --beta1 $beta1 --beta2 $beta2 --alg ppm --load_path $expname --num_workers 0 --val_freq 1 --extra_steps $num_step --max_epoch 800 --ema $ema --random_seed 0
        fi

  done

