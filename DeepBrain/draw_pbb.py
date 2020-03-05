import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import os
import shutil
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import datetime, time

model_type = "80_20_hard"
pbb_path = "./test_detector/results/"+model_type+"/bbox/"
clean_path = "/home/dwu/data/DZH_DATA/Sample3D/"
save_dir = "./draw_pbb/"

def write_dicom(pixel_array,filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelRepresentation = 1
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    minn = (np.max(pixel_array))
    maxx = (np.min(pixel_array))
    pixel_array = (pixel_array-minn).astype(float)/(maxx-minn).astype(float)*256
    # print(np.max(pixel_array))
    # print(np.min(pixel_array))
    # print(pixel_array[0])
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
	shutil.rmtree(save_dir)
	os.makedirs(save_dir)

idxs = []
for file in os.listdir(pbb_path):
	if "_pbb.npy" in file:
		idxs.append(file.replace("_pbb.npy",""))

for idx in idxs:
	print(idx)
	clean = np.load(clean_path+idx+"_clean.npy")[0]
	label = np.load(clean_path+idx+"_label.npy")
	pbb = np.load(pbb_path+idx+"_pbb.npy")
	
	save_idx_dir = "./draw_pbb/"+str(idx)+"/"
	if not os.path.exists(save_idx_dir):
	    os.makedirs(save_idx_dir)
	else:
	    shutil.rmtree(save_idx_dir)
	    os.makedirs(save_idx_dir)

	s = clean.shape
	i = 0

	for i in range(s[0]):
		idi = clean[i,:,:]
		write_dicom(idi,save_idx_dir+str(i)+".dcm")

	for pbbx in pbb:
		print(pbbx)
		p = pbbx[0]
		if p >= 0.9:
			x = int(pbbx[3])
			y = int(pbbx[2])
			z = int(pbbx[1])
			d = int(pbbx[4])

			for delta in range(-d/2,d/2):
				if z+delta>=s[0] or z+delta<0:
					continue
				test_clean = clean[z+delta,:,:]
				test_clean[y-d,x-d:x+d] = 1000
				test_clean[y+d,x-d:x+d] = 1000
				test_clean[y-d:y+d,x-d] = 1000
				test_clean[y-d:y+d,x+d] = 1000
				clean[z+delta,:,:] = test_clean
				write_dicom(test_clean,save_idx_dir+str(z+delta)+".dcm")