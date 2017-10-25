#for k in 100 1000 5000
#for k in 100 500 1000
#do
#    sbatch --job-name=dnhyper${k} script_sbatch.sh ${k}
#done

k=500
#for i in 0 $(seq 2 30)
#for i in 1
for i in 10 22 27
do
    sbatch --job-name=dhmons${k} script_sbatch_monserver_om.sh ${k} expEachAgain${i} 27017 ${i}
    #sbatch --job-name=dhmons${k} script_sbatch_monserver_om.sh ${k} expEach${i} 27017 ${i}
done

#for k in $(seq 1 300)
#for k in $(seq 1 30)
for k in $(seq 1 100)
do
    sbatch --job-name=dhmonc${k} script_sbatch_monclient_om.sh 27017 test_db
done
