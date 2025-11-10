#!/bin/bash

#SBATCH --job-name=pre_treino
#SBATCH --nodes=1
#SBATCH --output="../job_logs/%j.out"
#SBATCH --gpus=1

srun singularity run --nv \
    -B/pgeoprj,/pgeoprj2,/pgeodsv,/pgeo,/tatu,/scr01 \
    /tatu/container_images/basica_experimentacao.sif \
    /tatu/venv/ewab/bin/python train_and_eval.py --config configs/pretrain_config.yaml