import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import csv
from glob import glob
import traceback
import random
from PIL import Image
import math

from project_config import *

def acc(pbb, lbb):
    tp = []
    fp = []
    fn = []
   
    l_flag = np.zeros((len(lbb),), np.int32)
    
    for l in lbb:
    	flag = 0
        x = l[2]
        y = l[1]
        z = l[0]
        diameter_mm = l[3]
        radiusSquared = pow((diameter_mm), 2.0)
        for p in pbb:
        	x2 = p[3]
        	y2 = p[2]
        	z2 = p[1]
        	dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
        	if dist < radiusSquared:
        		tp.append(p)
        		flag = 1
        	else:
        		fp.append(p)
        if flag == 0:
        	fn.append(l)
    return tp, fp, fn

def write_csv(path):

    csv_path = path.replace("bbox/","CSVFILES/")
    l = os.listdir(path)
    srslst = []
    for i in l:
        if "_lbb.npy" in i:
            srslst.append(i.replace("_lbb.npy",""))
    print(srslst)

    dataframe = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm'])
    false_data = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm'])

    print("total:",len(srslst))
    for showid in range(len(srslst)):
        ctlab = np.load(path+srslst[showid]+'_lbb.npy')
        ctpre = np.load(path+srslst[showid]+'_pbb.npy')

        tp, fp, fn = acc(pbb=ctpre,lbb=ctlab)

        # for i in range(len(ctlab)):
        #     new_id = {'seriesuid':srslst[showid],'coordX':ctlab[i][2],'coordY':ctlab[i][1],'coordZ':ctlab[i][0],'diameter_mm':ctlab[i][3]}
        #     dataframe = dataframe.append(new_id,ignore_index=True)
        for i in range(len(tp)):
            new_id = {'seriesuid':srslst[showid],'coordX':tp[i][3],'coordY':tp[i][2],'coordZ':tp[i][1],'diameter_mm':tp[i][4]}
            dataframe = dataframe.append(new_id,ignore_index=True)
        # for i in range(len(fn)):
        #     new_id = {'seriesuid':srslst[showid],'coordX':fn[i][2],'coordY':fn[i][1],'coordZ':fn[i][0],'diameter_mm':fn[i][3]}
        #     dataframe = dataframe.append(new_id,ignore_index=True)
        for i in range(len(fp)):
            new_false = {'seriesuid':srslst[showid],'coordX':fp[i][3],'coordY':fp[i][2],'coordZ':fp[i][1],'diameter_mm':fp[i][4]}
            false_data = false_data.append(new_false,ignore_index=True)
        print(showid)
    print(dataframe)
    dataframe.to_csv(csv_path+"annotations.csv",index=False,sep=',',mode='w+')
    print(false_data)
    false_data.to_csv(csv_path+"candidates.csv",index=False,sep=',',mode='w+')

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def extract_real_cubic(path,annatation_file,plot_output_path,normalization_output_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: annatation_file:            the annatation csv file,contains every nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(plot ),every nodule end up withs three size
      @param:normalization_output_path:   the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(after normalization)
    '''
    print("begin to process real nodules...")
    l = os.listdir(path)
    srslst = []
    for i in l:
        if "_lbb.npy" in i:
            srslst.append(i.replace("_lbb.npy",""))
    df_node = pd.read_csv(annatation_file)
    print(len(df_node['seriesuid']))

    for i in range(len(df_node['seriesuid'])):
        real_id = df_node['seriesuid'][i]
        img_file = path + str(real_id) + "_clean.npy"

        img_array = np.load(img_file)[0]
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane

        img_array = img_array.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z

        node_x = int(df_node["coordX"][i])
        node_y = int(df_node["coordY"][i])
        node_z = int(df_node["coordZ"][i])
        node_d = int(df_node["diameter_mm"][i])

        nodule_pos_str = str(node_x)+"_"+str(node_y)+"_"+str(node_z)+"_"+str(node_d)
        # every nodules saved into size of 20x20x6,30x30x10,40x40x26
        imgs1 = np.ndarray([20,20,6],dtype=np.float32)
        imgs2 = np.ndarray([30,30,10],dtype=np.float32)
        imgs3 = np.ndarray([40,40,26],dtype=np.float32)
        center = np.array([node_x, node_y, node_z])   # nodule center
        v_center = np.rint(center)  # nodule center in voxel space (still x,y,z ordering)
        try:
            # these following imgs saves for plot
            imgs1[:,:,:]=img_array[int(v_center[0]-10):int(v_center[0]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[2]-3):int(v_center[2]+3)]
            imgs2[:,:,:]=img_array[int(v_center[0]-15):int(v_center[0]+15),int(v_center[1]-15):int(v_center[1]+15),int(v_center[2]-5):int(v_center[2]+5)]
            imgs3[:,:,:]=img_array[int(v_center[0]-20):int(v_center[0]+20),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-13):int(v_center[2]+13)]
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_real_size20x20.npy"),imgs1)
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_real_size30x30.npy"),imgs2)
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_real_size40x40.npy"),imgs3)
            # print("nodules %s from image %s extracted finished!..."%(node_idx,str(file_name)))
            # these following are the standard data as input of CNN

            # truncate_hu(imgs1)
            # truncate_hu(imgs2)
            # truncate_hu(imgs3)

            imgs1 = normalazation(imgs1)
            imgs2 = normalazation(imgs2)
            imgs3 = normalazation(imgs3)

            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_real_size20x20.npy"),imgs1)
            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_real_size30x30.npy"),imgs2)
            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_real_size40x40.npy"),imgs3)
            #print("normalization finished!..." )

        except Exception as e:
            print(" process images %s error..."%str(real_id))
            print(Exception,":",e)
            traceback.print_exc()

def extract_fake_cubic(dcim_path,annatation_file,plot_output_path,normalization_output_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: annatation_file:            the candidate csv file,contains every **fake** nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(plot ),every nodule end up withs three size
      @param:normalization_output_path:   the save path of extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file(after normalization)
    '''
    print("begin to process fake nodules...")
    l = os.listdir(path)
    srslst = []
    for i in l:
        if "_lbb.npy" in i:
            srslst.append(i.replace("_lbb.npy",""))
    df_node = pd.read_csv(annatation_file)
    print(len(df_node['seriesuid']))
    for i in range(len(df_node['seriesuid'])):
        real_id = df_node['seriesuid'][i]
        img_file = path + str(real_id) + "_clean.npy"

        img_array = np.load(img_file)[0]
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane

        img_array = img_array.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z

        s = img_array.shape

        node_x = int(df_node["coordX"][i])
        node_y = int(df_node["coordY"][i])
        node_z = int(df_node["coordZ"][i])
        node_d = int(df_node["diameter_mm"][i])

        if node_x-20<=0 or node_x+20>=s[0]:
        	continue
        if node_y-20<=0 or node_y+20>=s[1]:
        	continue
        if node_z-13<=0 or node_z+13>=s[2]:
        	continue
        	
        nodule_pos_str = str(node_x)+"_"+str(node_y)+"_"+str(node_z)+"_"+str(node_d)
        # every nodules saved into size of 20x20x6,30x30x10,40x40x26
        imgs1 = np.ndarray([20,20,6],dtype=np.float32)
        imgs2 = np.ndarray([30,30,10],dtype=np.float32)
        imgs3 = np.ndarray([40,40,26],dtype=np.float32)
        center = np.array([node_x, node_y, node_z])   # nodule center
        v_center = np.rint(center)  # nodule center in voxel space (still x,y,z ordering)
        try:
            # these following imgs saves for plot
            imgs1[:,:,:]=img_array[int(v_center[0]-10):int(v_center[0]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[2]-3):int(v_center[2]+3)]
            imgs2[:,:,:]=img_array[int(v_center[0]-15):int(v_center[0]+15),int(v_center[1]-15):int(v_center[1]+15),int(v_center[2]-5):int(v_center[2]+5)]
            imgs3[:,:,:]=img_array[int(v_center[0]-20):int(v_center[0]+20),int(v_center[1]-20):int(v_center[1]+20),int(v_center[2]-13):int(v_center[2]+13)]
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_fake_size20x20.npy"),imgs1)
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_fake_size30x30.npy"),imgs2)
            np.save(os.path.join(plot_output_path,str(real_id)+"_images_"+nodule_pos_str+"_fake_size40x40.npy"),imgs3)
            #print("nodules %s from image %s extracted finished!..."%(node_idx,str(file_name)))
            # these following are the standard data as input of CNN

            # truncate_hu(imgs1)
            # truncate_hu(imgs2)
            # truncate_hu(imgs3)

            imgs1 = normalazation(imgs1)
            imgs2 = normalazation(imgs2)
            imgs3 = normalazation(imgs3)

            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_fake_size20x20.npy"),imgs1)
            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_fake_size30x30.npy"),imgs2)
            np.save(os.path.join(normalization_output_path, str(real_id)+"_images_"+nodule_pos_str+"_fake_size40x40.npy"),imgs3)
            #print("normalization finished!..." )

        except Exception as e:
            print(" process images %s error..."%str(real_id))
            print(Exception,":",e)
            traceback.print_exc()

def check_nan(path):
    '''
     a function to check if there is nan value in current npy file path
    :param path:
    :return:
    '''
    for file in os.listdir(path):
        array = np.load(os.path.join(path,file))
        a = array[np.isnan(array)]
        if len(a)>0:
            print("file is nan :  ",file )
            print(a)

# LUNA2016 data prepare ,first step: truncate HU to -1000 to 400
def truncate_hu(image_array):
    image_array[image_array > 400] = 0
    image_array[image_array <-1000] = 0

# LUNA2016 dataprepare ,second step: normalzation the HU
def normalazation(image_array):
    max = 1000#image_array.max()
    min = -1000#image_array.min()
    # print("min",min,"max",max)
    image_array = (image_array-min).astype(float)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = 0#image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def search(path, word):
    '''
       find filename match keyword from path
    :param path:  path search from
    :param word:  keyword should be matched
    :return:
    '''
    filelist = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and word in filename:
            filelist.append(fp)
        elif os.path.isdir(fp):
            search(fp, word)
    return filelist

def get_all_filename(path,size):
    list_real = search(path, 'real_size' + str(size) + "x" + str(size))
    list_fake = search(path, 'fake_size' + str(size) + "x" + str(size))
    return list_real+list_fake

def get_test_batch(files):
    '''
    prepare every batch file data and label test
    :param path:
    :return:
    '''
    batch_array = []
    batch_label = []
    for npy in files:
        try:
            arr = np.load(npy)
            arr = arr.transpose(2, 1, 0)  #
            batch_array.append(arr)
            if 'real_' in npy.split("/")[-1]:
                batch_label.append([0, 1])
            elif 'fake_' in npy.split("/")[-1]:
                batch_label.append([1, 0])
        except Exception as e:
            print("file not exists! %s" % npy)
            batch_array.append(batch_array[-1])  # some nodule process error leading nonexistent of the file, using the last file copy to fill
            print(e.message)

    return np.array(batch_array), np.array(batch_label)

def get_train_batch(batch_filename,tag="20x20"):
    '''
    prepare every batch file data and label train
    :param batch_filename:
    :return:
    '''
    batch_array = []
    batch_label = []
    for npy in batch_filename:
        try:
            arr = np.load(npy)
            arr = arr.transpose(2,1,0)   #
            batch_array.append(arr)
            if 'real_' in  npy.split("/")[-1]:
                batch_label.append([0,1])
            elif 'fake_' in npy.split("/")[-1]:
                batch_label.append([1, 0])
        except Exception  as e:
            print("file not exists! %s"%npy)
            batch_array.append(batch_array[-1])  # some nodule process error leading nonexistent of the file, using the last file copy to fill

    return np.array(batch_array),np.array(batch_label)

def angle_transpose(file,degree,flag_string):
    '''
     @param file : a npy file which store all information of one cubic
     @param degree: how many degree will the image be transposed,90,180,270 are OK
     @flag_string:  which tag will be added to the filename after transposed
    '''
    array = np.load(file)
    array = array.transpose(2, 1, 0)  # from x,y,z to z,y,x
    newarr = np.zeros(array.shape,dtype=np.float32)
    for depth in range(array.shape[0]):
        jpg = array[depth]
        jpg.reshape((jpg.shape[0],jpg.shape[1],1))
        img = Image.fromarray(jpg)
        out = img.rotate(degree)
        newarr[depth,:,:] = np.array(out).reshape(array.shape[1],-1)[:,:]
    newarr = newarr.transpose(2,1,0)
    np.save(file.replace(".npy",flag_string+".npy"),newarr)

if __name__ =='__main__':

	# Get related information from the pbb file generated by the detector and write it to csv.
    write_csv(base_dir+"bbox/")

    # Extract three different cubes from the original CT images / preprocessing the data.
    path = "/home/dwu/data/DZH_DATA/Sample3D/"
    print("extracting image into %s"%normalazation_output_path)
    extract_real_cubic(path, annatation_file, plot_output_path, normalazation_output_path)
    extract_fake_cubic(path, candidate_file, plot_output_path, normalazation_output_path)

    print("finished!...")

    files = [os.path.join(normalazation_output_path,x) for x in os.listdir(normalazation_output_path)]
    print("how many files in normalzation path:  ",len(files))
    real_files = [m for m in files if "real" in m]
    print("how many files in real path:  ", len(real_files))
    # for file in real_files:
    #     angle_transpose(file,90,"_leftright")
    #     angle_transpose(file, 180, "_updown")
    #     angle_transpose(file, 270, "_diagonal")
    print("Final step done...")