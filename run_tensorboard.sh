#!/usr/bin/env bash
#SBATCH -t 8:00:00               # max runtime is 8 hours
#SBATCH --mem=8GB
#SBATCH --gres=gpu:0
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o Tensorboard-%A.out

# INSTRUCTIONS

# read port and ipaddr from Tensorboard-jobid.out, and then:
# ssh -L PORT:IPADDR:PORT username@V.vectorinstitute.ai


source activate sinkhorn

ipnport=21345 # TODO set your port here
echo ipnport="$2"

ipnip=$(hostname -i)
echo ipnip=$ipnip

tensorboard --logdir="$1" --port="$2" --bind_all
