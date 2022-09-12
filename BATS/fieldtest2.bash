#!/bin/bash -l

# Job name:
#SBATCH --job-name=bat2fieldtest2
#
# Partition:
#SBATCH --partition=lr6
#
# Account:
#SBATCH --account=lr_esd2
#
# qos:
#SBATCH --qos=condo_esd2
#
# Wall clock time:
#SBATCH --time=200:00:00
#
# Node count
#SBATCH --nodes=4
#
# Node feature
#SBATCH --constrain=lr6_cas


module load python
conda activate /global/home/users/hchen8/.conda/envs/pg
python fieldtimelapse.py >& mypy2.out