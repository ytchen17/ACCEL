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



onn_out_type = 'shiftRight_phaseNoise_0.26pi'


data_dict = sio.loadmat('./onn_output/onn_output_' + onn_out_type +'.mat')

train_data = np.array(data_dict["onn_output_train"]).astype(np.float32)
train_label = mnist.train.labels[0:2000, :]

test_data = np.array(data_dict["onn_output_test"]).astype(np.float32)
test_label = mnist.test.labels[0:1000, :]

print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
print("Load data done.")

train_data_rotate = np.zeros((2000,1024))
test_data_rotate = np.zeros((1000,1024))


w = 32
h = 32
(cX, cY) = (w // 2, h // 2)

rot_deg = 5

M = cv2.getRotationMatrix2D((cX, cY), rot_deg, 1.0)

for k in range(2000):
    img = train_data[k,:]
    img = np.resize(img,[32,32])
    rotated = cv2.warpAffine(img, M, (w, h))
    train_data_rotate[k, :] = np.resize(rotated, [1024])

for m in range(1000):
    img = test_data[m,:]
    img = np.resize(img,[32,32])
    rotated = cv2.warpAffine(img, M, (w, h))
    test_data_rotate[m, :] = np.resize(rotated, [1024])

data_dict = {'onn_output_train': train_data_rotate, 'onn_output_test': test_data_rotate}

sio.savemat('./onn_output/onn_output_' + onn_out_type + '_rotate-' + str(rot_deg) +'.mat', data_dict)

