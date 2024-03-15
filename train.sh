#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=03:55:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu
#SBATCH -o ./out/slurm-%j.out # STDOUT



module load CUDA
python better_resnet/main.py --lr 0.1 --db_p 0.1 --db_size 3 --dw 0.1 --db_sync False --filename_small None --filename_big ckpt_acc95.75_e197_dbs3_dbp0.1_dw0.0.pth
