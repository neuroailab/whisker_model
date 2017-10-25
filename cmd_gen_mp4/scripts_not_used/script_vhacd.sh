#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 10
#SBATCH --mem=100000
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_vhacd_%j.out

source activate env_torch_2
ssh -f -N -L 22334:localhost:22334 chengxuz@dicarlo5.mit.edu
python cmd_vhacd.py --startIndx ${1} --lenIndx ${2}
