#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=03:55:00
#SBATCH --account=plgcrossdistillphd-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu
#SBATCH -o ./out/slurm-%j.out # STDOUT
#SBATCH --array=1-10

module load CUDA
dw=$(echo "scale=1; $SLURM_ARRAY_TASK_ID / 10" | bc)
    python better_resnet/main.py --lr 0.1 --db_p 0.0 --big_drop 0.1 --db_size 7  --dw 1.0 --filename_small None --filename_big ckpt_acc78.98_e191_dbs7_dbp0.0_dw0.0.pth --dist_loss USKD
