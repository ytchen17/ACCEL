import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import os
import numpy as np
import pickle
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import scipy.io as sio

import argparse

import matplotlib.pyplot as plt 

import cv2



onn_out_type = 'noShift_phaseNoise_0.26pi'


data_dict = sio.loadmat('./onn_output/onn_output_' + onn_out_type +'.mat')

train_data = np.array(data_dict["onn_output_train"]).astype(np.float32)
train_label = mnist.train.labels[0:2000, :]

test_data = np.array(data_dict["onn_output_test"]).astype(np.float32)
test_label = mnist.test.labels[0:1000, :]

print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
print("Load data done.")

train_data_shift = np.zeros((2000,1024))
test_data_shift = np.zeros((1000,1024))

img_shift = np.zeros((32,32))


for k in range(2000):
    img = train_data[k,:]
    img = np.resize(img,[32,32])
    img_shift[:, 1:32] = img[:, 0:31]
    train_data_shift[k, :] = np.resize(img_shift, [1024])

for m in range(1000):
    img = test_data[m,:]
    img = np.resize(img,[32,32])
    img_shift[:, 1:32] = img[:, 0:31]
    test_data_shift[m, :] = np.resize(img_shift, [1024])

data_dict = {'onn_output_train': train_data_shift, 'onn_output_test': test_data_shift}

sio.savemat('./onn_output/onn_output_shiftRight_phaseNoise_0.26pi.mat', data_dict)

