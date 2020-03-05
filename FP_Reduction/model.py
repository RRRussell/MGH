# -*- coding:utf-8 -*-
'''
 the idea of this script came from LUNA2016 champion paper.
 This model conmposed of three network,namely Archi-1(size of 10x10x6),Archi-2(size of 30x30x10),Archi-3(size of 40x40x26)

'''

import tensorflow as tf
from tensorflow.python.ops import array_ops
from dzh_preprocess import get_train_batch,get_all_filename,get_test_batch
import random
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc 
import math 
import os
#import tensorflow.python.debug as tf_debug

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

class model(object):

    def __init__(self,learning_rate,keep_prob,batch_size,epoch):
        print(" network begin...")
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.epoch = epoch

        self.cubic_shape = [[6, 20, 20], [10, 30, 30], [26, 40, 40]]

    def archi_1(self,input,keep_prob):
        with tf.name_scope("Archi-1"):
            # input size is batch_sizex20x20x6
            # 5x5x3 is the kernel size of conv1, 1 is the input depth, 64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64],stddev=0.001),dtype=tf.float32,name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01,shape=[64]),dtype=tf.float32,name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input,w_conv1,strides=[1,1,1,1,1],padding='VALID'),b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1, keep_prob)
            # print(out_conv1.shape)
            # max pooling ,pooling layer has no effect on the data size
            hidden_conv1 = tf.nn.max_pool3d(out_conv1,strides=[1,1,1,1,1],ksize=[1,1,1,1,1],padding='SAME')
            # print(hidden_conv1.shape)
            # after conv1 ,the output size is batch_sizex4x16x16x64 ([batch_size,in_deep,width,height,output_deep])
            w_conv2 = tf.Variable(tf.random_normal([3,5,5,64,64], stddev=0.001), dtype=tf.float32,name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1,1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)
            # print(out_conv2.shape)
            # after conv2 ,the output size is batch_sizex2x12x12x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([1,5,5,64,64], stddev=0.001), dtype=tf.float32, name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1,1], padding='VALID'),b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)
            # print(out_conv3.shape)
            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])
            # after conv3, the output size is batch_sizex2x8x8x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3,[-1,64*8*8*2])
            w_fc1 = tf.Variable(tf.random_normal([64*8*8*2,150],stddev=0.001),name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3,w_fc1),tf.constant(0.001,shape=[150])))
            out_fc1 = tf.nn.dropout(out_fc1,keep_prob)
            # print(out_fc1.shape)
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])
            w_fc2 = tf.Variable(tf.random_normal([150, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)
            # print(out_fc2.shape)
            out_sm = tf.nn.softmax(out_fc2)

            # Two values are returned here, one that has not passed the softmax
            # and one that has passed the softmax, and different outputs will be selected later as needed.
            return out_fc2, out_sm

    def archi_2(self, input, keep_prob):
        with tf.name_scope("Archi-2"):
            # input size is batch_sizex30x30x10
            # 5x5x3 is the kernel size of conv1, 1 is the input depth, 64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3,5,5,1,64], stddev=0.001), dtype=tf.float32, name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input, w_conv1, strides=[1,1,1,1,1], padding='VALID'), b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1, keep_prob)
            # after conv1 ,the output size is batch_sizex8x26x26x64([batch_size,in_deep,width,height,output_deep])
            # print(out_conv1.shape)
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1,1,2,2,1], ksize=[1,1,2,2,1], padding='SAME')
            # after maxpooling ,the output size is batch_sizex8x13x13x64([batch_size,in_deep,width,height,output_deep])
            # print(hidden_conv1.shape)
            w_conv2 = tf.Variable(tf.random_normal([3,5,5,64,64], stddev=0.001), dtype=tf.float32, name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1,1,1,1,1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)
            # print(out_conv2.shape)
            # after conv2 ,the output size is batch_sizex6x9x9x64([batch_size,in_deep,width,height,output_deep])
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 64, 64], stddev=0.001), dtype=tf.float32,name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1,1,1,1,1], padding='VALID'), b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)
            # print(out_conv3.shape)
            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])
            # after conv3, the output size is batch_sizex4x5x5x64([batch_size,in_deep,width,height,output_deep])
            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3, [-1, 64 * 5 * 5 * 4])
            w_fc1 = tf.Variable(tf.random_normal([64 * 5 * 5* 4, 250], stddev=0.001), name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), tf.constant(0.001, shape=[250])))
            out_fc1 = tf.nn.dropout(out_fc1, keep_prob)
            # print(out_fc1.shape)
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([250, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)
            # print(out_fc2.shape)
            out_sm = tf.nn.softmax(out_fc2)

            return out_fc2, out_sm

    def archi_3(self, input, keep_prob):
        with tf.name_scope("Archi-3"):
            # input size is batch_sizex40x40x26
            # 5x5x3 is the kernel size of conv1, 1 is the input depth, 64 is the number output channel
            w_conv1 = tf.Variable(tf.random_normal([3, 5, 5, 1, 64], stddev=0.001), dtype=tf.float32, name='w_conv1')
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv1')
            out_conv1 = tf.nn.relu(tf.add(tf.nn.conv3d(input, w_conv1, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv1))
            out_conv1 = tf.nn.dropout(out_conv1, keep_prob)
            # after conv1 ,the output size is batch_sizex24x36x36x64([batch_size,in_deep,width,height,output_deep])
            # print(out_conv1.shape)
            hidden_conv1 = tf.nn.max_pool3d(out_conv1, strides=[1, 2, 2, 2, 1], ksize=[1, 2, 2, 2, 1], padding='SAME')
            # after maxpooling ,the output size is batch_sizex12x18x18x64([batch_size,in_deep,width,height,output_deep])
            # print(hidden_conv1.shape)
            w_conv2 = tf.Variable(tf.random_normal([3, 5, 5, 64, 64], stddev=0.001), dtype=tf.float32, name='w_conv2')
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv2')
            out_conv2 = tf.nn.relu(tf.add(tf.nn.conv3d(hidden_conv1, w_conv2, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv2))
            out_conv2 = tf.nn.dropout(out_conv2, keep_prob)
            # after conv2 ,the output size is batch_sizex10x14x14x64([batch_size,in_deep,width,height,output_deep])
            # print(out_conv2.shape)
            w_conv3 = tf.Variable(tf.random_normal([3, 5, 5, 64, 64], stddev=0.001), dtype=tf.float32,name='w_conv3')
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]), dtype=tf.float32, name='b_conv3')
            out_conv3 = tf.nn.relu(tf.add(tf.nn.conv3d(out_conv2, w_conv3, strides=[1, 1, 1, 1, 1], padding='VALID'), b_conv3))
            out_conv3 = tf.nn.dropout(out_conv3, keep_prob)
            # after conv3, the output size is batch_sizex8x10x10x64([batch_size,in_deep,width,height,output_deep])
            # print(out_conv3.shape)
            out_conv3_shape = tf.shape(out_conv3)
            tf.summary.scalar('out_conv3_shape', out_conv3_shape[0])

            # all feature map flatten to one dimension vector,this vector will be much long
            out_conv3 = tf.reshape(out_conv3, [-1, 64 * 10 * 10 * 8])
            w_fc1 = tf.Variable(tf.random_normal([64 * 10 * 10 * 8, 250], stddev=0.001), name='w_fc1')
            out_fc1 = tf.nn.relu(tf.add(tf.matmul(out_conv3, w_fc1), tf.constant(0.001, shape=[250])))
            out_fc1 = tf.nn.dropout(out_fc1, keep_prob)
            # print(out_fc1.shape)
            out_fc1_shape = tf.shape(out_fc1)
            tf.summary.scalar('out_fc1_shape', out_fc1_shape[0])

            w_fc2 = tf.Variable(tf.random_normal([250, 2], stddev=0.001), name='w_fc2')
            out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.001, shape=[2])))
            out_fc2 = tf.nn.dropout(out_fc2, keep_prob)
            # print(out_fc2.shape)
            out_sm = tf.nn.softmax(out_fc2)

            return out_fc2, out_sm

    def inference(self,npy_path,test_path,model_index,train_flag=True):

        all_filenames = get_all_filename(npy_path,self.cubic_shape[model_index][1])
        all_filenumbers = len(all_filenames)
        print("file size is : ",all_filenumbers)
        k = int(0.8*all_filenumbers)
        random.shuffle(all_filenames)
        train_filenames = all_filenames[:k]
        vaild_filenames = all_filenames[k:]
        # how many time should one epoch should loop to feed all data
        times = int(len(train_filenames) / self.batch_size)
        if (len(train_filenames) % self.batch_size) != 0:
            times = times + 1

        # keep_prob used for dropout
        keep_prob = tf.placeholder(tf.float32,name="keep_prob")
        # take placeholder as input
        x = tf.placeholder(tf.float32, [None, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2]],name="input_x")
        x_image = tf.reshape(x, [-1, self.cubic_shape[model_index][0], self.cubic_shape[model_index][1], self.cubic_shape[model_index][2], 1])
        if model_index == 0:
            net_out, softmax_out = self.archi_1(x_image,keep_prob)
        elif model_index == 1:
            net_out, softmax_out = self.archi_2(x_image,keep_prob)
        elif model_index == 2:
            net_out, softmax_out = self.archi_3(x_image,keep_prob)

        saver = tf.train.Saver(max_to_keep=10000)  # default to save all variable,save mode or restore from path
        tf.add_to_collection('net_out', net_out)
        tf.add_to_collection('softmax_out', softmax_out)

        if train_flag:
            # softmax layer
            real_label = tf.placeholder(tf.float32, [None, 2], name="label")
            # net_loss = focal_loss(prediction_tensor = net_out, target_tensor = real_label)
            # print("net_out",net_out)
            cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=real_label))

            net_loss = tf.reduce_mean(cross_entropy)

            # train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(net_loss)
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=net_loss)

            correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(real_label, 1))
            accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.add_to_collection('accruacy', accruacy)

            merged = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train_writer = tf.summary.FileWriter('./tensorboard-'+str(model_index+1)+'/', sess.graph)
                # loop epoches
                for i in range(self.epoch):
                    epoch_start =time.time()
                    #  the data will be shuffled by every epoch
                    random.shuffle(train_filenames)
                    total_loss=0
                    # train
                    for t in range(times):
                        batch_files = train_filenames[t*self.batch_size:(t+1)*self.batch_size]
                        batch_data, batch_label = get_train_batch(batch_files)
                        # print("len",len(batch_files))
                        feed_dict = {x: batch_data, real_label: batch_label,
                                     keep_prob: self.keep_prob}
                        _, summary, l, nnn = sess.run([train_step, merged, net_loss, net_out],feed_dict=feed_dict)
                        total_loss = total_loss+l

                        train_writer.add_summary(summary, i)

                        saver.save(sess, './ckpt-'+str(model_index+1)+'/archi-'+str(model_index+1), global_step=i + 1)

                    print("epoch ",i)
                    print("loss ",total_loss/times)

                    # vaild
                    if i%10==0:
                        vaild_batch,vaild_label = get_test_batch(vaild_filenames)
                        vaild_dict = {x: vaild_batch, real_label: vaild_label, keep_prob:self.keep_prob}
                        acc_vaild,loss = sess.run([accruacy,net_loss],feed_dict=vaild_dict)
                        with open('./archi-'+str(model_index+1)+'.txt','a+') as fw:#追加
                            fw.write('Epoch: '+str(i)+'\n')
                            fw.write('accuracy  is %f' % acc_vaild+'\n')
                            fw.write("loss is "+str(loss)+'\n')

                    epoch_end = time.time()

                    print(" epoch %d time consumed %f seconds"%(i,(epoch_end-epoch_start)))