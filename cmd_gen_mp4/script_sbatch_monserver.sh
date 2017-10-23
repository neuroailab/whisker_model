#!/bin/bash
#SBATCH -p yamins
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/users/chengxuz/slurm_output/hyper_monserver_%j.out

source ~/.tunnel
python cmd_hyperopt.py --neval ${1} --usemongo 1 --expname ${2} --portn ${3} --indxsta ${4}
