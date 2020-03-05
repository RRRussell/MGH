#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node007
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J aneurysm_train

cd ./DeepBrain/
sh ./train_detector.sh