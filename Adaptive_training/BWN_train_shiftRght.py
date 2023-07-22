import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import os
import numpy as np
import pickle
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import scipy.io as sio
import h5py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noise_scale", type = float, default= 0, help = "")
parser.add_argument("--onn_noise", type = float, default = 0, help = "")
parser.add_argument("--bwn_noise", type = float, default = 0, help = "")
parser.add_argument("--bias", type = float, default = 1.0, help = "")
parser.add_argument("--GPU", type = str, default= '0', help = "the GPU number used")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
tf_config = tf.ConfigProto() 
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=tf_config)


start_lr=1e-3
lr_decay_step=1200
lr_decay_rate = 0.9

BATCH_SIZE = 100

onn_out_type = 'shiftRight'
parameter_name =  onn_out_type + '_start-lr_' + str(start_lr) + '_step_' + str(lr_decay_step) + '_rate_' + str(lr_decay_rate)


sess = tf.InteractiveSession()
ckpt_dir = './ckpt_bwn/'
ckpt_restore_dir = './ckpt_bwn/'
restore = False
Train = True
global_steps = tf.Variable(0,trainable=False)


###
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def sigmoid_binarize(x,k):
    y = tf.sigmoid(k*x)
    y = y*2.0 - 1.0
    return y



'''prepare data'''

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

num_class = 10

# data

data_dict = sio.loadmat("./onn_output/onn_output_" + onn_out_type +".mat")
train_data = np.array(data_dict["onn_output_train"]).astype(np.float32)
train_label = mnist_data.train.labels[0:2000, :]

test_data = np.array(data_dict["onn_output_test"]).astype(np.float32)
test_label = mnist_data.test.labels[0:1000, :]


print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
print("Load data done.")


train_data_label = np.concatenate((train_data, train_label), axis = 1)


x = tf.placeholder("float", shape=[None, 1024]) #img,resized to 1024
y_ = tf.placeholder("float", shape=[None, 10]) #label
k = tf.compat.v1.placeholder("float")

x_input = x

x_input = x_input * args.bias
x_input = tf.add(x_input, tf.random_normal(shape=(BATCH_SIZE, 32*32), mean=0.0, stddev = args.onn_noise))
x_input = tf.nn.relu(x_input)


W_fc1 = weight_variable([1024, 10])
W_fc1 = tf.clip_by_value(W_fc1,-2.5,2.5)
bin_W_fc1 = binarize(W_fc1)
#bin_W_fc1 = sigmoid_binarize(W_fc1, k)
bin_W_fc1_test = binarize(W_fc1)

h_fc1 = tf.matmul(x_input, bin_W_fc1)
h_fc1_test = tf.matmul(x_input, bin_W_fc1_test)


h_fc1 = tf.add(h_fc1, tf.random_normal(shape=(BATCH_SIZE, 10), mean=0.0, stddev = args.bwn_noise))
h_fc1_test = tf.add(h_fc1_test, tf.random_normal(shape=(BATCH_SIZE, 10), mean=0.0, stddev = args.bwn_noise))

h_fc1 = h_fc1[:,0:10]
h_fc1_test = h_fc1_test[:,0:10]

y = tf.nn.softmax(h_fc1)
y_test = tf.nn.softmax(h_fc1_test)


cross_entropy = -tf.reduce_sum(y_*tf.log(y))
lr = tf.train.exponential_decay(start_lr, global_steps, lr_decay_step, lr_decay_rate, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy,global_step=global_steps)
correct_prediction = tf.equal(tf.argmax(y_test,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver(max_to_keep=1)

sess.run(tf.global_variables_initializer())

if restore==True:
  ckpt_model = tf.train.latest_checkpoint(ckpt_restore_dir)
  saver.restore(sess, ckpt_model)
  print ('Model restore done!')


acc_max = 0.0
acc_log = np.zeros((201,1))

count_num = 0
test_batch_num = int(1000 / BATCH_SIZE)
for batch_num in range(test_batch_num):
  data_temp = test_data[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]
  label_temp = test_label[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]

  acc_batch_test = accuracy.eval(feed_dict={x: data_temp, y_: label_temp, k:1000})
  count_num += acc_batch_test*BATCH_SIZE

test_accuracy = count_num/1000
acc_max = test_accuracy

print ('acc_max:', acc_max, "\t epoch:", -1)


acc_log[0] = test_accuracy

if Train==True:
  for i in range(200):
    if i < 20:
      coef_tmp = 1
    elif i < 40:
      coef_tmp = 10
    elif i < 60:
      coef_tmp = 100
    else:
      coef_tmp = 1000


    np.random.shuffle(train_data_label)

    train_batch_num = int(2000 / BATCH_SIZE)
    count_num = 0
    for batch_num in range(train_batch_num):
      batch = train_data_label[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]
      data_temp = batch[:, :1024]
      label_temp = batch[:, 1024:]

      train_step.run(feed_dict={x: data_temp, y_: label_temp, k: coef_tmp})

      acc_batch_train = accuracy.eval(feed_dict={x: data_temp, y_: label_temp, k: coef_tmp})
      count_num += acc_batch_train*BATCH_SIZE
    train_accuracy = count_num/2000
    # print("step %d, training accuracy %g"%(i, train_accuracy))

    test_batch_num = int(1000 / BATCH_SIZE)
    count_num = 0
    for batch_num in range(test_batch_num):
      data_temp = test_data[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]
      label_temp = test_label[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]

      acc_batch_test = accuracy.eval(feed_dict={x: data_temp, y_: label_temp, k: coef_tmp})
      count_num += acc_batch_test*BATCH_SIZE
    test_accuracy = count_num/1000


    if test_accuracy>acc_max:
      os.system('rm ' + "output_bwn/" + parameter_name + '_test_*_acc_%03.4f.mat' % (acc_max))
      os.system('rm ' + "output_bwn/" + parameter_name + '_acc_log_%03.4f.mat' % (acc_max))

      acc_max = test_accuracy
      saver.save(sess, os.path.join(ckpt_dir, parameter_name, 'test_%d_acc_%03.4f' % (i, acc_max)),global_step=i)
      print ('Model save done!\t acc_max:', acc_max, "\t epoch:", i)

      bin_W_fc1_val, h_fc1_val, x_input_val =sess.run(\
                      [bin_W_fc1, h_fc1, x_input],\
                      feed_dict={x: data_temp, y_: label_temp, k:1000})
      data_dict = {'bin_W_fc1':bin_W_fc1_val, 'h_fc1': h_fc1_val, 'input': x_input_val, 'test_acc': test_accuracy}
      sio.savemat("output_bwn/" + parameter_name + '_test_%d_acc_%03.4f.mat' % (i,acc_max), data_dict)

    if i%2 == 0:
      print("epoch:", i, "\t learning_rate:", sess.run(lr), "\t train_acc:", train_accuracy,\
            "\t test accuracy:%g \t acc_max:%g"%(test_accuracy, acc_max), '\t global_steps:', sess.run(global_steps))

    acc_log[i+1] = test_accuracy
    sio.savemat("./output_bwn/" + parameter_name + "_acc_log_%03.4f.mat" % (acc_max), {'acc_log':acc_log})
