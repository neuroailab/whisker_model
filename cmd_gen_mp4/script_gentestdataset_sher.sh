#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_gendata_%j.out

#python cmd_hdf5.py --pathhdf5 /scratch/users/chengxuz/barrel/barrel_relat_files/testdataset --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --objindx ${1} --generatemode 3 --testmode 2 --smallolen 4 --smallplen 3 --bigsamnum ${2} --hdf5suff ${3} --randseed ${4} --checkmode 1
#python cmd_hdf5.py --pathhdf5 /scratch/users/chengxuz/barrel/barrel_relat_files/testdataset_diver --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --objindx ${1} --generatemode 3 --testmode 2 --smallolen 1 --smallplen 1 --bigsamnum ${2} --hdf5suff ${3} --randseed ${4} --checkmode 1
python cmd_hdf5.py --pathhdf5 /scratch/users/chengxuz/barrel/barrel_relat_files/testdataset_control${5} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --objindx ${1} --generatemode 3 --testmode 2 --smallolen 1 --smallplen 1 --bigsamnum ${2} --hdf5suff ${3} --randseed ${4} --checkmode 1 --whichcontrol ${5}
