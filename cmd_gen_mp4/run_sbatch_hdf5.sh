:'
for objindx in 0 1
#for objindx in 0
do
    for oindxsta in $(seq 0 14)
    #for oindxsta in 0
    do
        for spindxsta in $(seq 0 13)
        #for spindxsta in 0
        do
            #sbatch script_sbatch_hdf5_om.sh ${objindx} ${oindxsta} ${spindxsta}
            sh run_check_hdf5.sh ${objindx} ${oindxsta} ${spindxsta}
        done
    done
done
'

#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 1 --scindxlen 1 --oindxsta ${2} --pindxsta ${3}
#python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s_val --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 3 --scindxlen 3 --oindxsta ${2} --pindxsta ${3} --generatemode 2 --testmode 2


:'
for objindx in 0 1
#for objindx in 0
do
    for pindxsta in 1 $(seq 12 19)
    #for pindxsta in 1
    do
        for scindxsta in $(seq 0 5)
        #for scindxsta in 0
        do
            for oindxsta in $(seq 0 3 14)
            #for oindxsta in 0
            do
                sbatch script_sbatch_hdf5_om.sh ${objindx} ${pindxsta} ${scindxsta} ${oindxsta}
            done
        done
    done
done
'

#data_path1=/scratch/users/chengxuz/barrel/bullet3/data/teddy2_VHACD_CHs.obj
#data_path2=/scratch/users/chengxuz/barrel/bullet3/data/duck_vhacd.obj
data_path1=/om/user/chengxuz/barrel/bullet3/data/teddy2_VHACD_CHs.obj
data_path2=/om/user/chengxuz/barrel/bullet3/data/duck_vhacd.obj

#bigsamnum=4
bigsamnum=96

for seed in $(seq 0 124)
#for seed in 0
#for seed in $(seq 251 500)
#for seed in $(seq 0 250)
#for seed in $(seq 130 150)
#for seed in 0
do
    #for control in 1 2
    #for control in 3 4
    #for control in 5
    for control in 6
    do
        #sbatch script_gentestdataset_sher.sh ${data_path1} ${bigsamnum} teddy_${seed} ${seed} ${control}
        #sbatch script_sbatch_hdf5_om.sh ${data_path1} ${bigsamnum} teddy_${seed} ${seed} ${control}
        sbatch script_sbatch_hdf5_om.sh ${data_path2} ${bigsamnum} duck_${seed} ${seed} ${control}
        #sbatch script_gentestdataset_sher.sh ${data_path1} ${bigsamnum} teddy_${seed} ${seed}
        #sbatch script_gentestdataset_sher.sh ${data_path2} ${bigsamnum} duck_${seed} ${seed}
    done
done

:'
for objindx in 0 1
#for objindx in 0
do
    #for pindxsta in 1 $(seq 12 19)
    for pindxsta in $(seq 0 3 35)
    #for pindxsta in 0
    do
        for scindxsta in $(seq 0 5)
        #for scindxsta in 0
        do
            for oindxsta in $(seq 0 5 14)
            #for oindxsta in 0
            do
                sbatch script_sbatch_hdf5_om.sh ${objindx} ${pindxsta} ${scindxsta} ${oindxsta}
                #sbatch script_gentestdataset_dis_sher.sh ${objindx} ${pindxsta} ${scindxsta} ${oindxsta}
            done
        done
    done
done
'
