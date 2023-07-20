import tensorflow as tf
import numpy as np
import mask_modulation_model_noise as mmm
import data_generation as dg
import scipy.io as sio
import os
import bwn
import argparse
import scipy.misc
import matplotlib.pyplot as plt

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

sess = tf.InteractiveSession()
ckpt_dir = './ckpt/bwn_noise_0_onn-noise_0_bias_1.0_GPU_0/'
restore = False
Train = True


onn_output_quantize = False
loss_kl = False
BATCH_SIZE = 100
MASK_ROW = 600
MASK_COL = 600
PAD = 50
MASK_PIXEL_NUM = MASK_ROW * MASK_COL
OBJECT_ROW = 28
OBJECT_COL = 28
OBJECT_PIXEL_NUM = OBJECT_ROW * OBJECT_COL
OBJECT_UPSAMPLING = 16
sub_type = '122'

mask_col_WOpad = MASK_COL - PAD*2
mask_row_WOpad = MASK_ROW - PAD*2


parameter_name = 'bwn_noise_' + str(args.bwn_noise) + '_onn-noise_' + str(args.onn_noise) + '_bias_' + str(args.bias)


def onn_subsampling_np(input,dim):  
    output = scipy.misc.imresize(input, (dim,dim))

    return output 

lr_decay_step=1500
# start_lr=5e-3
start_lr=1e-3
global_steps = tf.Variable(0,trainable=False)

if __name__ == '__main__':
    ModelNum = 0
    print('Network Model: Linear Real')
    print('')
    MODEL = 'init'
    LENS_NUM = 2
    MASK_NUMBER = 1
    SBN = 0
    LENS_f = 1000
    MASK_MASK_DISTANCE = 3000

    TOTAL_MASK_LAYERS = MASK_NUMBER*(LENS_NUM - 1)
    SAVING_PATH = './test_output/'

    x = tf.placeholder(tf.complex64, shape=(BATCH_SIZE, MASK_PIXEL_NUM))
    y_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10))
    k = tf.compat.v1.placeholder("float")

    phase_noise_dict = sio.loadmat('./onn_output/mask_phase_noise_0.26pi.mat')
    phase_noise = phase_noise_dict['phase_noise']

    #onn
    onn_measurement, onn_mask_phase, onn_mask_amp = mmm.inference_init(x, MASK_NUMBER, LENS_NUM, SBN, LENS_f, MASK_MASK_DISTANCE, phase_noise)
    onn_input2 = tf.reshape(onn_measurement,[-1,MASK_COL,MASK_ROW,1])
    onn_input2 = onn_input2[:,PAD:PAD+mask_row_WOpad,PAD:PAD+mask_row_WOpad,:]
    
    #subsampling

    if sub_type == '800':
        onn_input2 = tf.nn.avg_pool(onn_input2,[1,25,25,1],[1,25,25,1],'SAME')
    elif sub_type == '122':
        onn_input2 = onn_input2[:,64:436,64:436,:]
        onn_input2 = tf.image.resize_images(onn_input2,(32,32), method=0)


    onn_input2 = tf.reshape(onn_input2,[-1,32*32])
    
    if onn_output_quantize == True:
       g = tf.get_default_graph()
       round_level = 370 - 1
       with g.gradient_override_map({'Round': 'Identity'}):
           onn_input2 = tf.round(onn_input2 * round_level) / round_level
  
    onn_input2 = tf.nn.l2_normalize(onn_input2, dim=1)
    bwn_input = onn_input2 * args.bias
    bwn_input = tf.add(bwn_input, tf.random_normal(shape=(BATCH_SIZE, 32*32), mean=0.0, stddev = args.onn_noise))
    bwn_input = tf.nn.relu(bwn_input)


    # bwn
    W_fc1 = bwn.weight_variable([1024, 16])
    W_fc1 = tf.clip_by_value(W_fc1,-1.5,1.5)
    bin_W_fc1 = bwn.sign_binarize(W_fc1)
    
    h_fc1 = tf.matmul(bwn_input, bin_W_fc1)

    h_fc1 = tf.add(h_fc1, tf.random_normal(shape=(BATCH_SIZE, 16), mean=0.0, stddev = args.bwn_noise))

    h_fc1 = h_fc1[:,0:10]
    y = tf.nn.softmax(h_fc1)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_gt,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #training 
    saver = tf.train.Saver()       
    sess.run(tf.global_variables_initializer())    
    max_acc = 0.0
 
    # if restore==True:
    ckpt_model = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, ckpt_model)
    print ('Model restore done!')
    
    '''testing'''
    print("---------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------Inference Begin-----------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------------------------")
    print("bias:", args.bias, "\tonn-noise:", args.onn_noise, "\tbwn-noise", args.bwn_noise)


    onn_input_train = np.zeros([2000, 784])
    onn_output_train = np.zeros([2000, 1024])
    onn_input_test = np.zeros([1000, 784])
    onn_output_test = np.zeros([1000, 1024])


    train_data = dg.mnist_data.train.images[0:2000, :]
    train_label = dg.mnist_data.train.labels[0:2000, :]
    train_data_label = np.concatenate((train_data, train_label), axis = 1)

    test_data = dg.mnist_data.test.images[0:1000, :]
    test_label = dg.mnist_data.test.labels[0:1000, :]
    test_data_label = np.concatenate((test_data, test_label), axis = 1)


    train_batch_num = int(2000/BATCH_SIZE)
    count_num = 0
    for batch_num in range(train_batch_num):
        batch = train_data_label[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]
        train_amp = batch[:, :784]
        input_label = batch[:, 784:]


        train_amp = train_amp + np.random.normal(loc=0.0, scale=args.noise_scale,size=(BATCH_SIZE,OBJECT_PIXEL_NUM))
        train_amp[np.where(train_amp<0)] = 0
        input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        input_amp = dg.object_zero_boundary(train_amp)
        input_phase = dg.object_zero_boundary(input_phase)
        input_amp = dg.object_upsampling_1(input_amp)
        input_phase = dg.object_upsampling_1(input_phase)      
        input_amp = dg.object_padding(input_amp)
        input_phase = dg.object_padding(input_phase) * 1.999 * np.pi
    
        input = input_amp * np.cos(input_phase) + 1j * input_amp * np.sin(input_phase)   

        acc_batch, onn_output_temp = sess.run([accuracy, onn_input2], feed_dict={x: input, y_gt: input_label, k:10000})
        onn_output_train[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE] = onn_output_temp
        onn_input_train[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE] = train_amp

        count_num += acc_batch*BATCH_SIZE
        print("epoch:", 0, "\t train_batch_num:", batch_num, "\t train_acc:", acc_batch)

    train_acc = count_num/2000
    print ("train accuracy %g"%(train_acc))    

    test_batch_num = int(1000/BATCH_SIZE)
    count_num = 0
    for batch_num in range(test_batch_num):
        batch = test_data_label[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE, :]
        test_amp = batch[:, :784]
        input_label = batch[:, 784:]

        test_amp = test_amp + np.random.normal(loc=0.0, scale=args.noise_scale,size=(BATCH_SIZE,OBJECT_PIXEL_NUM))
        test_amp[np.where(test_amp<0)] = 0
        input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        input_amp = dg.object_zero_boundary(test_amp)
        input_phase = dg.object_zero_boundary(input_phase)
        input_amp = dg.object_upsampling_1(input_amp)
        input_phase = dg.object_upsampling_1(input_phase)      
        input_amp = dg.object_padding(input_amp)
        input_phase = dg.object_padding(input_phase) * 1.999 * np.pi
    
        input = input_amp * np.cos(input_phase) + 1j * input_amp * np.sin(input_phase)   

        acc_batch, onn_output_temp = sess.run([accuracy, onn_input2], feed_dict={x: input, y_gt: input_label, k:10000})
        onn_output_test[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE] = onn_output_temp
        onn_input_test[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE] = test_amp

        count_num += acc_batch*BATCH_SIZE
        print("epoch:", 0, "\t test_batch_num:", batch_num, "\t test_acc:", acc_batch)

    test_acc = count_num/1000
    print ("test accuracy %g"%(test_acc))
    print("---------------------------------------------------------------------------------------------------------------------")
    print("--------------------------------------------------Inference End------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------------------------")
    

    data_dict = {'onn_output_train': onn_output_train, 'onn_output_test': onn_output_test}

    sio.savemat('./onn_output/onn_output_noShift_phaseNoise_0.26pi.mat', data_dict)
