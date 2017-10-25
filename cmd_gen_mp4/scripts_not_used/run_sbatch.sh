#for k in 100 1000 5000
#for k in 100 500 1000
#do
#    sbatch --job-name=dnhyper${k} script_sbatch.sh ${k}
#done

#k=500
#k=1000
#sbatch --job-name=dhmons${k} script_sbatch_monserver.sh ${k} exp4
#sbatch --job-name=dhmons${k} script_sbatch_monserver.sh ${k} exp12
#sleep 5
#for k in $(seq 1 15)
#do
#    sbatch --job-name=dhmonc${k} script_sbatch_monclient.sh 23333 test_db
#done

#len=1025
#len=410
#for k in $(seq 0 ${len} 51228)
#for k in 35260 14760
#for k in 14760
#do
#    sbatch --job-name=vhacd${k} script_vhacd.sh ${k} ${len}
#done

#len_data=25
#len_data=10
#len_data=1
len_data=20

#bigsamnum=12
bigsamnum=2

len_tfr=100

#module load tensorflow/0.12.1
#module load anaconda/anaconda.4.2.0.python2.7

#for k in $(seq 0 ${len_data} 9981)
#for k in 9160
#for k in 5554
#for k in 4039
#for k in $(seq ${len_data} ${len_data} 9981)
#for k in $(seq 6250 ${len_data} 9981)
#for k in $(seq 5065 5069)
#for k in $(seq 0 ${len_data} 399)
#for k in 0
for k in $(seq 0 ${len_tfr} 1711)
#for k in $(seq 1 116)
#for k in $(seq 0 116)
#for k in 1
#for k in 4119
#for k in $(seq 0 ${len_data} 399)
do
    #sbatch --job-name=dataset${k} script_gendataset.sh ${k} ${len_data} ${bigsamnum}
    #sbatch --qos=use-everything --job-name=dataset${k} script_gendataset.sh ${k} ${len_data} ${bigsamnum}
    #sbatch --job-name=dataset${k} script_gendataset_sher.sh ${k} ${len_data} ${bigsamnum}
    #sbatch --job-name=dataset${k} script_genvaldataset_sher.sh ${k} 1 ${bigsamnum}
    #sbatch --job-name=dataset${k} script_genvaldataset_sher.sh ${k} ${len_data} ${bigsamnum}
    #sbatch --job-name=dataset${k} script_gendataset_sher.sh ${k} 1 ${bigsamnum}
    #sbatch --job-name=dataset${k} script_gendataset.sh ${k} ${len_data} ${bigsamnum}
    #sbatch --job-name=tfrecs${k} script_to_tfrecs.sh ${k} ${len_data}
    #sbatch --job-name=tfrecs${k} script_to_tfrecs_sher.sh ${k} ${len_data}
    #sbatch --job-name=tfrecs${k} script_to_tfr_bycat.sh ${k}
    #sbatch --job-name=tfrecs${k} script_to_tfr_bycat.sh ${k}
    #python cmd_to_tfr_bycat.py --catsta ${k} --catlen 117
    #sbatch --job-name=tfrecs${k} script_to_tfr_bycat_om.sh ${k}

    #sbatch --job-name=compute${k} script_computestat.sh ${k} ${len_data} Data_force
    #sbatch --job-name=compute${k} script_computestat.sh ${k} ${len_data} Data_torque
    #sbatch --job-name=compute${k} script_computestat_sher.sh ${k} ${len_data} Data_force
    #sbatch --job-name=compute${k} script_computestat_sher.sh ${k} ${len_data} Data_torque
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} Data_force
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} Data_torque
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} category
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} speed
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} position
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} orn
    #sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} scale
    sbatch --job-name=tfrecs${k} script_rewrite_tfr.sh ${k} ${len_tfr} objid
done
