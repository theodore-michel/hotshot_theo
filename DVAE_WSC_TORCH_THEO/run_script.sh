#!/bin/bash
#SBATCH -A rcd@gpu
#SBATCH --job-name=DVAE
#SBATCH --output=DVAE.out
#SBATCH --error=DVAE.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=40:00:00
#SBATCH --qos=qos_gpu-t4


# cleans out modules loaded in interactive and inherited by default
module purge

# loading modules
module load pytorch-gpu/py3/1.7.0

# echo of launched commands
set -x

#DVAE with skip connections (WSC)
outname='DVAE_WSC_test'


# code execution
srun python -u run_train_model_torch.py --db_name ../../DATABASES/run_db_DVAE_PEGS_only  --train_nsamp 347600  --val_nsamp 102400 --test_nsamp 50000 --batch_size 512 --lr 0.001 --n_epochs 100 --log_interval 100 --ncomp 1 --skip_co [True,True,True] --check_interval 20 --model_outname $outname

