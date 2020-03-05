import os
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

path = "/home/dwu/AneurysmDetection/dzh/FP_Reduction/train/"+"FP_DATA/cubic_npy/"

srslist = []

idx = 0
for file in os.listdir(path):
	f = np.load(path+file)
	s = f.shape
	test_clean = f[:,:,s[2]/2]
	scipy.misc.imsave("./crop/"+str(idx)+"_"+file+".png",test_clean)
	idx = idx+1