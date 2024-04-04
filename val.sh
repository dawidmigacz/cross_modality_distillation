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
python ./validate.py
