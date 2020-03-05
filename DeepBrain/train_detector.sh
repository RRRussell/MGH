#!/bin/bash
set -e

# python prepare.py
cd ./detector/
maxeps=10000

CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py --model dpn3d26 -b 16 --gpu 0,1,2,3  --epochs $maxeps --config config_training --save-dir "80_20_hard" 
