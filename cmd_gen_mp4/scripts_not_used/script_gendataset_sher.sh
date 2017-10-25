#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_gendata_%j.out

#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000

# Train dataset on sherlock
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000 --checkmode 0

# Val dataset
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5_val --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 20000 --checkmode 0

# Change to write info files
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000 --checkmode 2


# Change to write info files for validataion
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5_val --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 20000 --checkmode 2

# Regenerate the dataset
python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset2/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000
