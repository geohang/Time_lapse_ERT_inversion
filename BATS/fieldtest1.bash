#!/bin/bash -l

# Job name:
#SBATCH --job-name=bat2fieldtest1
#
# Partition:
#SBATCH --partition=lr6
#
# Account:
#SBATCH --account=lr_esd2
#
# qos:
#SBATCH --qos=lr_lowprio
#
# Wall clock time:
#SBATCH --time=20:00:00
#
# Node count
#SBATCH --nodes=1
#
# Node feature
#SBATCH --constrain=lr6_cas


module load python
conda activate /global/home/users/hchen8/.conda/envs/pg
python invtest1.py >& mypy1.out