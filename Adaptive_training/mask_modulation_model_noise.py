import tensorflow as tf

import os
import math
import numpy as np
import free_space_propagation as fsp
import initialization as init
import scipy.io as sio
from scipy.io import loadmat

mask_quantize = False
BATCH_SIZE = 100
MASK_ROW = 600
MASK_COL = 600
PAD = 50
MAX_STEPS = 180000
DISPLAY_STEPS = 1000
MASK_PIXEL_NUM = MASK_ROW * MASK_COL
BATCH_SHAPE = [BATCH_SIZE, MASK_ROW, MASK_COL]
OBJECT_ROW = 28
OBJECT_COL = 28
OBJECT_UPSAMPLING = 16
TEST_NUM = 1000
LOSS_FUNCTION = 'mse'
SENSOR_MOD = 1
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
MASK_PHASE_MODULATION = True
MASK_AMPLITUDE_MODULATION = False
MASK_INIT_TYPE = 'const'
MASK_INIT_VALUE = 0.0
MASK_PIXEL_SIZE = 3.0
REF_IDX = 1.4602
LAMDA = 532
MASK_LAYER_THICKNESS = 600
LENS_RADIUS = 3000
MEASUREMENT_MOD = 'square'
OBJECT_MASK_DISTANCE = 0
MASK_SENSOR_DISTANCE = 3000 #um


def mask_init():
    if MASK_INIT_TYPE == 'const':
        if MASK_PHASE_MODULATION is True:
            mask_phase_org = tf.Variable(tf.constant(MASK_INIT_VALUE, shape=[1, (MASK_COL-PAD*2)*(MASK_ROW-PAD*2)]), name='mask_phase')

            if mask_quantize == True:
                mask_phase_1 = tf.sigmoid(mask_phase_org)
                g = tf.get_default_graph()
                round_level = 256 - 1
                with g.gradient_override_map({'Round': 'Identity'}):
                    mask_phase = tf.round(mask_phase_1 * round_level) / round_level
                mask_phase = mask_phase * 2.0 * np.pi
                
            if mask_quantize == False:
                mask_phase = np.pi * 2.0 * tf.sigmoid(mask_phase_org*1.0)
            
        if MASK_AMPLITUDE_MODULATION is True:
            pass
        else:          
            mask_amp = tf.ones([1, MASK_PIXEL_NUM])
    elif MASK_INIT_TYPE == 'random':
        
        if MASK_PHASE_MODULATION is True:
            mask_phase = tf.Variable(tf.random_uniform((1, (MASK_COL-PAD*2)*(MASK_ROW-PAD*2)), 0, 1), name='mask_phase')
        else:
            pass
        if MASK_AMPLITUDE_MODULATION is True:
            pass
        else:          
            mask_amp = tf.ones([1, MASK_PIXEL_NUM])

    elif MASK_INIT_TYPE == 'load': # default: modulate phase and not modulate amplitute
        mask_amp = tf.ones([1, MASK_PIXEL_NUM])
        load_mask_phase = loadmat("./mask_phase_138_1199.mat")
        flag_ae = 'dec'
        name = flag_ae+'_mask_phase'
        mask_phase_trained = load_mask_phase[name]
        mask_phase = tf.Variable(mask_phase_trained)     
        

    else:
        pass
    
    paddings = [[0,0],[PAD,PAD],[PAD,PAD]]
    mask_phase = tf.reshape(mask_phase,[1,MASK_COL-PAD*2,MASK_ROW-PAD*2])
    mask_phase = tf.pad(mask_phase, paddings, mode="CONSTANT", name=None, constant_values=0)
    mask_phase = tf.reshape(mask_phase,[1,MASK_PIXEL_NUM])

    return mask_phase, mask_amp


def loss_crop(measurement, ground_truth):

    measurement = tf.reshape(measurement, BATCH_SHAPE)
    ground_truth = tf.reshape(ground_truth, BATCH_SHAPE)
    measurement = measurement[:, 25:-25, 25:-25]
    ground_truth = ground_truth[:, 25:-25, 25:-25]
    measurement = tf.reshape(measurement, (BATCH_SIZE, 10000))
    ground_truth = tf.reshape(ground_truth, (BATCH_SIZE, 10000))
    return measurement, ground_truth

def prop_through_lens(img, batch_shape, pixelsize, lamda, focal_length, radius, convunits=True):

    img_reshaped = tf.reshape(img, batch_shape)
    if convunits is True:
        lamda = lamda * 1e-9
        pixelsize = pixelsize * 1e-6
        focal_length = focal_length * 1e-6
        radius = radius * 1e-6
    Nx = pixelsize * np.arange((-np.ceil((batch_shape[1] - 1) / 2)), np.floor((batch_shape[1] - 1) / 2)+0.5)
    Ny = pixelsize * np.arange((-np.ceil((batch_shape[2] - 1) / 2)), np.floor((batch_shape[2] - 1) / 2)+0.5)
    [x,y] = np.meshgrid(Nx,Ny)
    np_P = ((x ** 2.0 + y ** 2.0) <= (radius ** 2.0))
    temp = -1j * math.pi / lamda / focal_length
    np_F = np.multiply(np_P, np.exp(temp * (x ** 2.0 + y ** 2.0)))
    F = tf.convert_to_tensor(np_F, dtype=tf.complex64)
    img_prop = tf.multiply(img_reshaped, F)
    img_prop_reshaped = tf.reshape(img_prop, [batch_shape[0], batch_shape[1]*batch_shape[2]])

    return img_prop_reshaped


def loss_function_softmax_ce(measurement, ground_truth):

    measurement_reshaped = tf.reshape(measurement, BATCH_SHAPE)

    if SENSOR_MOD == 1:
        cell_width = np.round(100 / 3)
        cell_width_center_line = np.round(100 / 4)
        shift = int(MASK_ROW/2) - 50
        image_size_nopadding = OBJECT_ROW * OBJECT_UPSAMPLING
        image_size = MASK_ROW
        # padding = (image_size - image_size_nopadding) / 2
        margin = np.round(cell_width * 0.30)
        margin_center_line = np.round(cell_width_center_line * 0.25)
        sensor_width = int(cell_width - 2 * margin - 1)
        sensor_width_center_line = int(cell_width_center_line - 2 * margin_center_line - 1)

        for i in range(BATCH_SIZE):

            for j in range(0, 3):
                sensor_start_x = int(j * cell_width + margin + 1) + shift
                sensor_start_y = int(margin + 1) + shift
                tmp = tf.reduce_mean(measurement_reshaped[i, sensor_start_y : sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width])
                if j == 0:
                    mean_measurement = tf.expand_dims(tmp, -1)
                else:
                    mean_measurement = tf.concat([mean_measurement, tf.expand_dims(tmp, -1)], -1)

            for j in range(3, 7):
                sensor_start_x = int((j - 3) * cell_width_center_line + margin_center_line + 1) + shift
                sensor_start_y = int(cell_width + margin + 1) + shift
                tmp = tf.reduce_mean(measurement_reshaped[i, sensor_start_y : sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width])
                mean_measurement = tf.concat([mean_measurement, tf.expand_dims(tmp, -1)], -1)

            for j in range(7, 10):
                sensor_start_x = int((j - 7) * cell_width + margin + 1) + shift
                sensor_start_y = int(2 * cell_width + margin + 1) + shift
                tmp = tf.reduce_mean(measurement_reshaped[i, sensor_start_y : sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width])
                mean_measurement = tf.concat([mean_measurement, tf.expand_dims(tmp, -1)], -1)

            tmp_loss = tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth[i, :], logits = mean_measurement), -1)
            if i == 0:
                loss = tmp_loss
            else:
                loss = tf.concat([loss, tmp_loss], -1)


    loss_mean = tf.reduce_mean(loss, name='softmax_ce')
    tf.summary.scalar('loss', loss_mean)

    return loss_mean

def loss_function_sigmoid_ce(measurement, ground_truth):

    loss = tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth, logits = measurement)

    loss_mean = tf.reduce_mean(loss, name='sigmoid_ce')
    tf.summary.scalar('loss', loss_mean)

    return loss_mean


def loss_function_mse(measurement, ground_truth):

    factor = tf.divide( tf.reduce_sum(tf.multiply(ground_truth, measurement)), tf.reduce_sum(tf.multiply(measurement, measurement)) )
    squared_deltas = tf.square(tf.square(measurement * factor - ground_truth))
    # squared_deltas = tf.square(measurement - ground_truth)

    loss = tf.reduce_sum(squared_deltas, reduction_indices=[1])
    loss_mean = tf.reduce_mean(loss, name='mse')

    tf.summary.scalar('loss', loss_mean)

    return loss_mean

def tv_loss_function(measurement):

    measurement = tf.reshape(measurement, BATCH_SHAPE)
    #measurement = measurement[:, 25:-25, 25:-25]
    pixel_dif1 = tf.subtract(measurement[:, 1:, :], measurement[:, :-1, :])
    pixel_dif2 = tf.subtract(measurement[:, :, 1:], measurement[:, :, :-1])
    sum_axis = [1, 2]
    tot_var = (tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) +
               tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis))
    tot_var_mean = tf.reduce_mean(tot_var, name='tv')

    #tot_var_mean = tf.nn.l2_loss(pixel_dif1) + tf.nn.l2_loss(pixel_dif2)

    return tot_var_mean


def training(loss):

    if OPTIMIZER == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    else:
        pass
    train_op = optimizer.minimize(loss)

    return train_op


def inference_init(img, MASK_NUMBER, LENS_NUM, SBN, LENS_f, MASK_MASK_DISTANCE, phase_noise):

    # First Layer
    with tf.name_scope('hidden1'):

        with tf.name_scope('propagation'):
            img_p = fsp.batch_propagate(img, BATCH_SHAPE, MASK_PIXEL_SIZE, REF_IDX, LAMDA, MASK_LAYER_THICKNESS, True, False, True)
            img_p_new = fsp.batch_propagate(img_p, BATCH_SHAPE, MASK_PIXEL_SIZE, 1.0, LAMDA, OBJECT_MASK_DISTANCE, True, False, True)

        with tf.name_scope('mask'):
            mask_phase, mask_amp = mask_init()
            mask_phase = mask_phase + phase_noise
            mask_phase = tf.clip_by_value(mask_phase, 0, np.pi * 2.0)
            mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
            save_mask_phase = mask_phase
            save_mask_amp = mask_amp

        hidden = tf.multiply(img_p_new, mask)
        hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, REF_IDX, LAMDA, MASK_LAYER_THICKNESS, True, False, True)
        if SBN == 1:
            hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 1.0, LAMDA, 100, True, False, True)
            hidden_amp = tf.abs(hidden)
            delta_phase = math.pi * hidden_amp / (1 + hidden_amp)
            hidden = tf.multiply(hidden, tf.complex(tf.cos(delta_phase), tf.sin(delta_phase)))
            hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 2.33, LAMDA, 1000, True, False, True)


    # Middle Layers
    for layer_num in range(2, MASK_NUMBER + 1):
        with tf.name_scope('hidden' + str(layer_num)):
            with tf.name_scope('propagation'):
                img_p_new = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 1.0, LAMDA, MASK_MASK_DISTANCE, True, False, True)

            with tf.name_scope('mask'):
                mask_phase, mask_amp = mask_init()
                mask_phase = mask_phase + phase_noise
                mask_phase = tf.clip_by_value(mask_phase, 0, np.pi * 2.0)
                mask = tf.complex(mask_amp * tf.cos(mask_phase), mask_amp * tf.sin(mask_phase))
                save_mask_phase = tf.concat([save_mask_phase, mask_phase], 0)
                save_mask_amp = tf.concat([save_mask_amp, mask_amp], 0)

            hidden = tf.multiply(img_p_new, mask)
            hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, REF_IDX, LAMDA, MASK_LAYER_THICKNESS, True, False, True)

        if SBN == 1:
            hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 1.0, LAMDA, 100, True, False, True)
            hidden_amp = tf.abs(hidden)
            delta_phase = math.pi * hidden_amp / (1 + hidden_amp)
            hidden = tf.multiply(hidden, tf.complex(tf.cos(delta_phase), tf.sin(delta_phase)))
            hidden = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 2.33, LAMDA, 1000, True, False, True)

    # Last Layer
    with tf.name_scope('last'):

        with tf.name_scope('propagation'):
            img_p_new = fsp.batch_propagate(hidden, BATCH_SHAPE, MASK_PIXEL_SIZE, 1.0, LAMDA, MASK_SENSOR_DISTANCE, True, False, True)

        with tf.name_scope('sensor'):
            if MEASUREMENT_MOD == 'abs':
                measurement =tf.abs(img_p_new)# tf.square()
            elif MEASUREMENT_MOD == 'square':
                measurement = tf.square(tf.abs(img_p_new))

    #measurement = tf.nn.l2_normalize(measurement, dim=1)

    return measurement, save_mask_phase, save_mask_amp
