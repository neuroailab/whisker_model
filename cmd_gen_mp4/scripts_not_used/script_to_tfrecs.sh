#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 1
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_tfrecs_%j.out

source activate env_torch_2
python cmd_to_tfrecords.py --objsta ${1} --objlen ${2}
