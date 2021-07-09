#!/bin/bash
#SBATCH -A rcd@gpu
#SBATCH --job-name=DVAE_wsc_plat
#SBATCH --output=DVAE_wsc_plat.out
#SBATCH --error=DVAE_wsc_plat.out
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
outname='DVAE_WSC_testing'


# code execution
srun python -u run_train_model_torch.py --db_name ../../DATABASES/run_db_test  --train_nsamp 25600  --val_nsamp 5120 --test_nsamp 2560 --batch_size 512 --lr 0.001 --n_epochs 100 --log_interval 100 --ncomp 1 --skip_co '111' --check_interval 25 --model_outname $outname --platform JZ --test_mode BEST
#srun python -u run_train_model_torch.py --db_name ../../DATABASES/run_db_test  --train_nsamp 3476  --val_nsamp 1024 --test_nsamp 50000 --batch_size 512 --lr 0.001 --n_epochs 3 --log_interval 100 --ncomp 1 --check_interval 20 --model_outname $outname --platform JZ --test_mode BEST
