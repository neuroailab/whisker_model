#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 1
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_tfrecs_%j.out

source activate env_torch_2
python cmd_to_tfr_bycat.py --catsta ${1} --catlen 1 --seedbas 0 --loaddir /om/user/chengxuz/Data/barrel_dataset2/raw_hdf5 --savedir /om/user/chengxuz/Data/barrel_dataset2/tfrecords --suffix om_strain
