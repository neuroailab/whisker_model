#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p yamins
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_tfrecs_%j.out

module load tensorflow/0.12.1
module load anaconda/anaconda.4.2.0.python2.7
#python cmd_to_tfrecords.py --objsta ${1} --objlen ${2} --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords --infodir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_info --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --seedbas 10000

# Val dataset
#python cmd_to_tfrecords.py --objsta ${1} --objlen ${2} --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords_val --infodir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_info_val --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5_val --seedbas 20000 --bigsamnum 2

python cmd_to_tfr_bycat.py --catsta ${1} --catlen 1
#python cmd_to_tfr_bycat.py --catsta ${1} --catlen 1 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset2/raw_val --suffix sval --bigsamnum 2 --seedbas 20000
