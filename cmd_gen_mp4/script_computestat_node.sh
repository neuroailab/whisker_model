python compute_stat.py --tfstart ${1} --tflen 100 --tfkey Data_force --saveprefix /home/chengxuz/ --tfdir /mnt/fs2/chengxuz/Data/whisker/tfrecs_all/tfrecords/

#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 1
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_computestat_%j.out

