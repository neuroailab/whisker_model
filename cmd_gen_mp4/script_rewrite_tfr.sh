#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 2
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/rewrite_totfrecs_%j.out

python rewrite_tfrecords.py --staindx ${1} --lenindx ${2} --key ${3}
