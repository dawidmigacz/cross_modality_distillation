#!/bin/bash

# Zmienna reprezentująca liczbę zadań
TASKS=10

# Ładowanie modułu CUDA
module load CUDA

# Pętla wykonująca zadania
for ((i=1; i<=TASKS; i++)); do
    dw=$(echo "scale=1; $i / 10" | bc)
    python better_resnet/main.py --lr 0.1 --db_p 0.1 --db_size 7 --db_sync True --dw 1.0 --filename_small None --filename_big ckpt_acc78.81_e198_dbs7_dbp0.1_dw0.0_sync.pth
done