#!/bin/bash
set -e

# python prepare.py
cd ./detector/
maxeps=10000

CUDA_VISIBLE_DEVICES="1,2,3" python main.py --model dpn3d26 -b 1 --save-dir "test" --epochs $maxeps --config config_training --resume "" 

#dpn3d26
#res18