# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from model import model
import numpy as np

from project_config import *

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

if __name__ =='__main__':
    print(" beigin...")
    model = model(learning_rate,keep_prob,batch_size,10000)
    model.inference(normalazation_output_path,test_path,2,True)







