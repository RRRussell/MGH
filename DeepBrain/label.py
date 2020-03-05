import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import os


path="./detector/crop/"
idxs = []
for file in os.listdir(path):
	if "crop" in file:
		idxs.append(file.replace("crop.npy",""))

for idx in idxs:
	print(idx)
	clean = np.load(path+idx+"crop.npy")[0]
	labelx = np.load(path+idx+"target.npy")
	s = clean.shape
	# i = 0
	# for labelx in label:
	print(labelx)
	test_clean = clean[int(labelx[0]),:,:]
	print(labelx)
	x = int(labelx[2])
	y = int(labelx[1])
	d = int(labelx[3])
	print(x,y)
	test_clean[y-d,x-d:x+d] = 1000
	test_clean[y+d,x-d:x+d] = 1000
	test_clean[y-d:y+d,x-d] = 1000
	test_clean[y-d:y+d,x+d] = 1000
	scipy.misc.imsave("./crop_label/"+idx+".png",test_clean)
	# i=i+1