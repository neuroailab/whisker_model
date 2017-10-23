#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_gendata_dis_%j.out

python cmd_hdf5.py --pathhdf5 /scratch/users/chengxuz/barrel/barrel_relat_files/testdataset_dis --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --objindx ${1} --pindxsta ${2} --scindxsta ${3} --oindxsta ${4} --oindxlen 5 --spindxlen 14 --generatemode 2 --testmode 2 --pindxlen 3
