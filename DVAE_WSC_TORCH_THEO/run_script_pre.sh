#!/bin/bash
#SBATCH -A rcd@cpu
#SBATCH --job-name=DVAE_pre
#SBATCH --output=DVAE_pre.out
#SBATCH --error=DVAE_pre.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t4 


# cleans out modules loaded in interactive and inherited by default
module purge

# loading modules
module load pytorch-gpu/py3/1.7.0

# echo of launched commands
set -x

outdbname="../DATABASES/run_db_test"

# code execution
srun python -u run_preprocess_databases.py --train_n_samp 347600  --val_n_samp 102400 --test_n_samp 50000 --output_db $outdbname --PEGS_db_path ../DATABASES/database_STF_duration_55_100_500k.hdf5 --NOISE_db_path ../DATABASES/noise_db8.hdf5 --ncomp 1

