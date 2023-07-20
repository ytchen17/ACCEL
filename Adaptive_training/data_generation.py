import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy as sci
from PIL import Image
import initialization as init
#from scipy.misc import imresize
from scipy import ndimage
from skimage.transform import resize

from skimage.morphology import skeletonize
from skimage.morphology import dilation, disk



# dict = init.import_parameters('./PARAMETERS1206.txt')
BATCH_SIZE = 100
MASK_ROW = 600
MASK_COL = 600
PAD = 50
LEARNING_RATE = 0.001
MAX_STEPS = 180000
DISPLAY_STEPS = 1000
MASK_PIXEL_NUM = MASK_ROW * MASK_COL
OBJECT_ROW = 28
OBJECT_COL = 28
TEST_NUM = 1000
LOSS_FUNCTION = 'mse'
SENSOR_MOD = 1
MASK_PIXEL_SIZE = 1
MASK_LAYER_THICKNESS = 100
LENS_RADIUS = 1000
MEASUREMENT_MOD = 'square'
APPLICATION = 'classification'

OBJECT_PHASE_INPUT = False
OBJECT_AMPLITUDE_INPUT = True
OBJECT_UPSAMPLING_1 = 16
OBJECT_UPSAMPLING_2 = 1
OBJECT_UPSAMPLING = 16
OBJECT_PIXEL_NUM = OBJECT_ROW * OBJECT_COL
OBJECT_PIXEL_SIZE = 1

TRAINING_DATA_TYPE = 'mnist'
TESTING_DATA_TYPE = 'mnist'
CAPTURED_DATA_PATH = 'captured_dataset'
SKELETON = False
SENSOR_MOD = 1
gt_mod = 800

g_index_in_epoch_train = 0
g_epochs_completed_train = 0
g_index_in_epoch_test = 0
g_epochs_completed_test = 0

if TRAINING_DATA_TYPE == 'mnist' or TESTING_DATA_TYPE == 'mnist':
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

if TRAINING_DATA_TYPE == 'fashion-mnist' or TESTING_DATA_TYPE == 'fashion-mnist':
    fashion_mnist_data = input_data.read_data_sets('data/fashion', one_hot=True)

if TRAINING_DATA_TYPE == 'captured':
    captured_training_data = init.object_read('/' + CAPTURED_DATA_PATH + '/training_dataset')

if TESTING_DATA_TYPE == 'captured':
    captured_testing_data = init.object_read('/' + CAPTURED_DATA_PATH + '/testing_dataset')

def sum_whole(onn_measurement2, dim_z):
    y = tf.reshape(onn_measurement2,[-1,2*dim_z,np.int32(MASK_PIXEL_NUM/(2*dim_z))])
    z_20 = tf.reduce_sum(y,2) 
    mu = z_20[:, :dim_z]
    sigma = z_20[:,dim_z:]

    return mu,sigma

def sum_175_140_block_dim10(onn_measurement2, dim_z):
    y = tf.reshape(onn_measurement2,[-1,MASK_COL,MASK_ROW])
    
    block = y[:,0:140, 0:175]
    block = tf.reshape(block,[-1,1,140*175])
    z_20 = tf.reduce_sum(block,2)  
    
    for i in range(5):
        for j in range(4):
            if (i==0)&(j==0) == False:
                block = y[:,i*140:(i+1)*140, j*175:(j+1)*175]
                block = tf.reshape(block,[-1,1,140*175])
                temp = tf.reduce_sum(block,2) 
                z_20 = tf.concat([z_20,temp],1)
                           
    mu = z_20[:, :dim_z]
    sigma = z_20[:,dim_z:]        
    return mu, sigma

def object_padding(input):

    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_COL * OBJECT_UPSAMPLING))
    input_padded = np.lib.pad(input, ((0, 0), (np.int32((MASK_ROW - OBJECT_ROW * OBJECT_UPSAMPLING) / 2), np.int32((MASK_COL - OBJECT_COL * OBJECT_UPSAMPLING) / 2)),
                                      (np.int32((MASK_ROW - OBJECT_ROW * OBJECT_UPSAMPLING) / 2), np.int32((MASK_COL - OBJECT_COL * OBJECT_UPSAMPLING) / 2))), 'edge')
    input_padded = np.reshape(input_padded, (BATCH_SIZE, MASK_PIXEL_NUM))
    return input_padded


def object_rotation(input, angle):

    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW, OBJECT_COL))
    for i in range(BATCH_SIZE):
        input[i, :, :] = sci.ndimage.interpolation.rotate(input[i, :, :], angle, reshape=False)
    input = np.reshape(input, (BATCH_SIZE, OBJECT_PIXEL_NUM))

    return input


def object_zero_boundary(input):

    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW, OBJECT_COL))
    for i in range(BATCH_SIZE):
        input[i, 0, :] = input[i, :, 0] = input[i, -1, :] = input[i, :, -1] = 0
    input = np.reshape(input, (BATCH_SIZE, OBJECT_PIXEL_NUM))

    return input

def object_zero_boundary_2(input):

    input = np.reshape(input, (BATCH_SIZE, MASK_ROW, MASK_ROW))
    for i in range(BATCH_SIZE):
        input[i, 0, :] = input[i, :, 0] = input[i, -1, :] = input[i, :, -1] = 0
    input = np.reshape(input, (BATCH_SIZE, MASK_ROW*MASK_ROW))

    return input


def object_upsampling_1(input):
    """
    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW, OBJECT_COL))
    output = np.zeros((BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_COL * OBJECT_UPSAMPLING))
    for i in range(BATCH_SIZE):
        for j in range(OBJECT_UPSAMPLING):
            for k in range(OBJECT_UPSAMPLING):
                output[i, j:OBJECT_ROW * OBJECT_UPSAMPLING:OBJECT_UPSAMPLING, k:OBJECT_COL * OBJECT_UPSAMPLING:OBJECT_UPSAMPLING] = input[i, :, :]
    output = np.reshape(output, (BATCH_SIZE, OBJECT_PIXEL_NUM * OBJECT_UPSAMPLING * OBJECT_UPSAMPLING))
    """
    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW, OBJECT_COL))
    output = np.zeros((BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING_1, OBJECT_COL * OBJECT_UPSAMPLING_1))
    for i in range(BATCH_SIZE):
        #output[i, :, :] = imresize(input[i, :, :], np.float32(OBJECT_UPSAMPLING_1), interp='bilinear', mode='F')
        output[i, :, :] = np.array(Image.fromarray(input[i, :, :]).resize((OBJECT_ROW * OBJECT_UPSAMPLING_1, OBJECT_ROW * OBJECT_UPSAMPLING_1)))
    output = np.reshape(output, (BATCH_SIZE, OBJECT_PIXEL_NUM * OBJECT_UPSAMPLING_1 * OBJECT_UPSAMPLING_1))

    return output

def object_upsampling_2(input):
    """
    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW, OBJECT_COL))
    output = np.zeros((BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_COL * OBJECT_UPSAMPLING))
    for i in range(BATCH_SIZE):
        for j in range(OBJECT_UPSAMPLING):
            for k in range(OBJECT_UPSAMPLING):
                output[i, j:OBJECT_ROW * OBJECT_UPSAMPLING:OBJECT_UPSAMPLING, k:OBJECT_COL * OBJECT_UPSAMPLING:OBJECT_UPSAMPLING] = input[i, :, :]
    output = np.reshape(output, (BATCH_SIZE, OBJECT_PIXEL_NUM * OBJECT_UPSAMPLING * OBJECT_UPSAMPLING))
    """
    input = np.reshape(input, (BATCH_SIZE, OBJECT_ROW*OBJECT_UPSAMPLING_1, OBJECT_COL*OBJECT_UPSAMPLING_1))
    output = np.zeros((BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_COL * OBJECT_UPSAMPLING))
    for i in range(BATCH_SIZE):
        #output[i, :, :] = imresize(input[i, :, :], np.float32(OBJECT_UPSAMPLING_2), interp='bilinear', mode='F')
        output[i, :, :] = np.array(Image.fromarray(input[i, :, :]).resize((OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_ROW * OBJECT_UPSAMPLING)))
        #output[i, :, :] = resize(input[i, :, :], (input[i, :, :].shape[0] * OBJECT_UPSAMPLING_2, input[i, :, :].shape[1] * OBJECT_UPSAMPLING_2), order=1, preserve_range='True')

        kernel = np.ones((2*OBJECT_UPSAMPLING_2, 2*OBJECT_UPSAMPLING_2))
        output[i, :, :]= ndimage.convolve(output[i, :, :], kernel, mode='constant', cval=0.0)
        if np.amax(output[i, :, :]) > 0:
            output[i, :, :] = output[i, :, :] / np.amax(output[i, :, :])

    output = np.reshape(output, (BATCH_SIZE, OBJECT_PIXEL_NUM * OBJECT_UPSAMPLING * OBJECT_UPSAMPLING))

    return output

def object_skeleton(input):
    (shape1, shape2)=input.shape
    obj_row_col = np.sqrt(shape2)
    input = np.reshape(input, (BATCH_SIZE, int(obj_row_col), int(obj_row_col)))
    output = np.zeros((BATCH_SIZE, int(obj_row_col), int(obj_row_col)))
    for i in range(BATCH_SIZE):
        input[i, :, :] = input[i, :, :] > 0.5
        output[i, :, :] = skeletonize(input[i, :, :])
    output =  np.reshape(output, (BATCH_SIZE, int(shape2)))
    return output

def object_next_batch_train(batch_size):

    global g_index_in_epoch_train
    global g_epochs_completed_train
    global captured_training_data

    """Return the next `batch_size` examples from this data set."""
    start = g_index_in_epoch_train
    # Shuffle for the first epoch
    if g_epochs_completed_train == 0 and start == 0:
        perm0 = np.arange(captured_training_data.shape[0])
        np.random.shuffle(perm0)
        captured_training_data = captured_training_data[perm0]
    # Go to the next epoch
    if start + batch_size > captured_training_data.shape[0]:
        # Finished epoch
        g_epochs_completed_train += 1
        # Get the rest examples in this epoch
        rest_num_examples = captured_training_data.shape[0] - start
        images_rest_part = captured_training_data[start:captured_training_data.shape[0]]
        # Shuffle the data
        perm = np.arange(captured_training_data.shape[0])
        np.random.shuffle(perm)
        captured_training_data = captured_training_data[perm]
        # Start next epoch
        start = 0
        g_index_in_epoch_train = batch_size - rest_num_examples
        end = g_index_in_epoch_train
        images_new_part = captured_training_data[start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0)
    else:
        g_index_in_epoch_train += batch_size
        end = g_index_in_epoch_train
        return captured_training_data[start:end]


def object_next_batch_test(batch_size):

    global g_index_in_epoch_test
    global g_epochs_completed_test
    global captured_testing_data

    """Return the next `batch_size` examples from this data set."""
    start = g_index_in_epoch_test
    # Shuffle for the first epoch
    if g_epochs_completed_test == 0 and start == 0:
        perm0 = np.arange(captured_testing_data.shape[0])
        np.random.shuffle(perm0)
        captured_testing_data = captured_testing_data[perm0]
    # Go to the next epoch
    if start + batch_size > captured_testing_data.shape[0]:
        # Finished epoch
        g_epochs_completed_test += 1
        # Get the rest examples in this epoch
        rest_num_examples = captured_testing_data.shape[0] - start
        images_rest_part = captured_testing_data[start:captured_testing_data.shape[0]]
        # Shuffle the data
        perm = np.arange(captured_testing_data.shape[0])
        np.random.shuffle(perm)
        captured_testing_data = captured_testing_data[perm]
        # Start next epoch
        start = 0
        g_index_in_epoch_test = batch_size - rest_num_examples
        end = g_index_in_epoch_test
        images_new_part = captured_testing_data[start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0)
    else:
        g_index_in_epoch_test += batch_size
        end = g_index_in_epoch_test
        return captured_testing_data[start:end]


def gt_generator_mnist(labels):

    #gt_labels = np.zeros((BATCH_SIZE, OBJECT_ROW * OBJECT_UPSAMPLING, OBJECT_COL * OBJECT_UPSAMPLING), dtype=np.float32)
    gt_labels = np.zeros((BATCH_SIZE, MASK_ROW, MASK_ROW), dtype=np.float32)

    if SENSOR_MOD == 1:
        cell_width = np.round(gt_mod/3)
        cell_width_center_line = np.round(gt_mod/4)
        shift = int(MASK_ROW/2) - int(gt_mod/2)
        #padding = (image_size - image_size_nopadding) / 2
        margin = np.round(cell_width*0.30)
        margin_center_line = np.round(cell_width_center_line*0.25)
        sensor_width = int(cell_width-2*margin-1)
        sensor_width_center_line = int(cell_width_center_line-2*margin_center_line-1)

        for i in range(BATCH_SIZE):
            #size_of_detector = 8 #in pixels
            for j in range(0, 3):
                sensor_start_x = int(j*cell_width+margin+1) + shift
                sensor_start_y = int(margin+1) + shift
                gt_labels[i, sensor_start_y: sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width] = labels[i,j]
            for j in range(3, 7):
                sensor_start_x = int((j-3)*cell_width_center_line+margin_center_line+1) + shift
                sensor_start_y = int(cell_width + margin+1) + shift
                gt_labels[i, sensor_start_y: sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width_center_line] = labels[i,j]
            for j in range(7, 10):
                sensor_start_x = int((j-7)*cell_width+margin+1) + shift
                sensor_start_y = int(2*cell_width + margin+1) + shift
                gt_labels[i, sensor_start_y: sensor_start_y + sensor_width, sensor_start_x: sensor_start_x + sensor_width] = labels[i,j]


    gt_labels_1 = labels

    gt_labels_final = np.reshape(gt_labels, (BATCH_SIZE,MASK_ROW*MASK_ROW))

    return gt_labels_final, gt_labels_1


def generate_data(datatype):

    if datatype == 'training':
        if TRAINING_DATA_TYPE == 'mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = mnist_data.train.next_batch(BATCH_SIZE)
                #input_amp = input_amp > 0.5
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = mnist_data.train.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TRAINING_DATA_TYPE == 'fashion-mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = fashion_mnist_data.train.next_batch(BATCH_SIZE)
                #input_amp = input_amp > 0.5
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = fashion_mnist_data.train.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TRAINING_DATA_TYPE == 'random':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_amp = object_zero_boundary(input_amp)
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_phase = object_zero_boundary(input_phase)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TRAINING_DATA_TYPE == 'captured':
            input_amp = object_next_batch_train(BATCH_SIZE)
            input_amp = object_zero_boundary(input_amp)
            input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

    elif datatype =='testing':
        if TESTING_DATA_TYPE == 'mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = mnist_data.test.next_batch(BATCH_SIZE)
                #input_amp = input_amp > 0.5
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = mnist_data.test.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

            #input_amp = object_rotation(input_amp, 45)
            #input_phase = object_rotation(input_phase, 45)

        elif TESTING_DATA_TYPE == 'fashion-mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = fashion_mnist_data.test.next_batch(BATCH_SIZE)
                #input_amp = input_amp > 0.5
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = fashion_mnist_data.test.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])


        elif TESTING_DATA_TYPE == 'random':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_amp = object_zero_boundary(input_amp)
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_phase = object_zero_boundary(input_phase)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TESTING_DATA_TYPE == 'captured':
            input_amp = object_next_batch_test(BATCH_SIZE)
            input_amp = object_zero_boundary(input_amp)
            input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

    elif datatype =='validation':
        if TESTING_DATA_TYPE == 'mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = mnist_data.validation.next_batch(BATCH_SIZE)
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = mnist_data.validation.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

            #input_amp = object_rotation(input_amp, 45)
            #input_phase = object_rotation(input_phase, 45)

        elif TESTING_DATA_TYPE == 'fashion-mnist':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp, input_amp_label = fashion_mnist_data.validation.next_batch(BATCH_SIZE)
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase, input_phase_label = fashion_mnist_data.validation.next_batch(BATCH_SIZE)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TESTING_DATA_TYPE == 'random':

            if OBJECT_AMPLITUDE_INPUT is True:
                input_amp = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_amp = object_zero_boundary(input_amp)
            else:
                input_amp = np.ones([BATCH_SIZE, OBJECT_PIXEL_NUM])
            if OBJECT_PHASE_INPUT is True:
                input_phase = np.random.rand(BATCH_SIZE, OBJECT_PIXEL_NUM)
                input_phase = object_zero_boundary(input_phase)
            else:
                input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

        elif TESTING_DATA_TYPE == 'captured':
            input_amp = object_next_batch_test(BATCH_SIZE)
            input_amp = object_zero_boundary(input_amp)
            input_phase = np.zeros([BATCH_SIZE, OBJECT_PIXEL_NUM])

    input_amp = object_zero_boundary(input_amp)
    input_phase = object_zero_boundary(input_phase)

    input_amp = object_upsampling_1(input_amp)
    input_phase = object_upsampling_1(input_phase)

    if SKELETON == 'true':
        input_amp = object_skeleton(input_amp)
        input_phase = object_skeleton(input_phase)

        input_amp = object_upsampling_2(input_amp)
        input_phase = object_upsampling_2(input_phase)

    input_amp = object_padding(input_amp)
    input_phase = object_padding(input_phase) * 1.999 * np.pi

#    input_amp = input_amp > 0.5

    input = input_amp * np.cos(input_phase) + 1j * input_amp * np.sin(input_phase)

    if APPLICATION == 'amplitude_imaging':
        gt = input_amp
    if APPLICATION == 'phase_imaging':
        gt = input_phase
    if APPLICATION == 'classification':
        if OBJECT_AMPLITUDE_INPUT is True:
            gt, gt_1 = gt_generator_mnist(input_amp_label)
            gt = object_zero_boundary_2(gt)
            #gt = object_padding(gt)
        if OBJECT_PHASE_INPUT is True:
            gt, gt_1 = gt_generator_mnist(input_phase_label)
            gt = object_zero_boundary_2(gt)
            #gt = object_padding(gt)

    return input, gt, gt_1
