import tensorflow as tf
from tensorflow.python.ops import array_ops
from dzh_preprocess import get_train_batch,get_all_filename,get_test_batch
import random
import numpy as np
import time
import math 
import pandas as pd
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

'''
If you want to use this test file, you need to modify the model_path and model names 
and rename the generated csv file. The generated file records the predicted probability of each bbox.
'''

model_path = '/home/dwu/AneurysmDetection/dzh/FP_Reduction/ckpt-3/'
data_path = "/home/dwu/AneurysmDetection/dzh/FP_Reduction/test/FP_DATA/cubic_normalization_npy/"
cubic_shape = [[6, 20, 20], [10, 30, 30], [26, 40, 40]]
model_index = 2
batch_size = 1
all_filenames = get_all_filename(data_path,cubic_shape[model_index][1])

with tf.Session() as sess:
	# Load graphs and corresponding parameters from already trained models
	new_saver = tf.train.import_meta_graph(model_path+"archi-3-450.meta")
	new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

	graph = tf.get_default_graph()

	accruacy = tf.get_collection('accruacy')[0]
	net_out = tf.get_collection('net_out')[0]
	softmax_out = tf.get_collection('softmax_out')[0]

	input_x = graph.get_operation_by_name('input_x').outputs[0]
	keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
	label = graph.get_operation_by_name('label').outputs[0]

	times = int(len(all_filenames) / batch_size)
	total_acc = 0
	total_file = []
	for t in range(times):
		batch_files = all_filenames[t*batch_size:(t+1)*batch_size]
		batch_data, batch_label = get_test_batch(batch_files)
		test_dict = {input_x: batch_data, label: batch_label, keep_prob: 1}
		acc_test,net_outt,softmax_outt = sess.run([accruacy,net_out,softmax_out],feed_dict=test_dict)
		total_acc = total_acc + acc_test
		total_file.append(batch_files)
		if t==0:
			total_net_outt = net_outt
			total_softmax_outt = softmax_outt
			total_label = batch_label
		else:
			total_net_outt = np.vstack((total_net_outt,net_outt))
			total_softmax_outt = np.vstack((total_softmax_outt,softmax_outt))
			total_label = np.vstack((total_label,batch_label))
	total_acc = total_acc/times

	save_dir_ = "./bbox/"
	if not os.path.exists(save_dir_):
		os.makedirs(save_dir_)
	else:
		import shutil
		shutil.rmtree(save_dir_)
		os.makedirs(save_dir_)

	# mrn, coordinates, radius and labels are recorded in the name of each file.
	# We integrate the predicted probability and this information into a csv file as the output of the model.
	dataframe = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm','probability'])
	for i in range(len(total_file)):
		f = total_file[i][0]
		name = f.split("/",9)[-1]
		s = name.split("_",7)
		mrn = s[0]
		x = s[2]
		y = s[3]
		z = s[4]
		d = s[5]
		p = total_softmax_outt[i][1]
		print(mrn,x,y,z,d,p)  
		new_id = {'seriesuid':mrn,'coordX':x,'coordY':y,'coordZ':z,'diameter_mm':d,'probability':p}
		dataframe = dataframe.append(new_id,ignore_index=True)
	csv_path = "./"
	dataframe.to_csv(csv_path+"DeepBrain_FP_3.csv",index=False,sep=',',mode='w+')