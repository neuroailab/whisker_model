#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=/om/user/chengxuz/slurm_out_all/hyper_server_%j.out

source activate env_torch
source ~/.tunnel
python cmd_hyperopt.py --neval ${1} --usemongo 1 --expname ${2} --portn ${3} --indxsta ${4}
