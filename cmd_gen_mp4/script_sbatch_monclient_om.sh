#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=/om/user/chengxuz/slurm_out_all/hyper_client_%j.out

source activate env_torch
source ~/.tunnel
# Export pythonpath
hyperopt-mongo-worker --mongo=localhost:${1}/${2} --poll-interval=1
