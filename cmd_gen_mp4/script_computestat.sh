#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 1
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_computestat_%j.out

source activate env_torch_2
python compute_stat.py --tfstart ${1} --tflen ${2} --tfkey ${3} --saveprefix /om/user/chengxuz/Data/barrel_dataset/statistics/${3}_
