#!/bin/bash
#SBATCH -p yamins
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/users/chengxuz/slurm_output/hyper_client_%j.out

source ~/.tunnel
source ~/.bashrc
hyperopt-mongo-worker --mongo=localhost:${1}/${2} --poll-interval=1
