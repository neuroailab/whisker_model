#!/bin/bash
#SBATCH -p yamins
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/users/chengxuz/slurm_output/hyper_%j.out

python cmd_hyperopt.py --neval ${1}
