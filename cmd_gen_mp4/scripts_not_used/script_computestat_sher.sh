#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 1
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_computestat_%j.out

module load tensorflow/0.12.1
module load anaconda/anaconda.4.2.0.python2.7

python compute_stat.py --tfstart ${1} --tflen ${2} --tfkey ${3} --saveprefix /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/statistic/${3}_ --tfdir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords/
