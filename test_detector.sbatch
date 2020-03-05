#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node008
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J aneurysm_test

cd ./DeepBrain/
sh ./test_detector.sh