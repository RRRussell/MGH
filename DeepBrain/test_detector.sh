#!/bin/bash
set -e

# python prepare.py
cd ./test_detector/
maxeps=10000

CUDA_VISIBLE_DEVICES="2,3" python main.py --model dpn3d26 -b 1 --gpu 2,3  --epochs $maxeps --config config_training --save-dir "80_20_hard" --resume "../detector/results/80_20_hard/1100.ckpt" 

#dpn3d26
#res18