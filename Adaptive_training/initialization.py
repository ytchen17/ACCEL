"""
dict = init.import_parameters('./PARAMETERS.txt')
CLASSIFICATION_THRESHOLD = float(dict['classification_threshold'])
APPLICATION = (dict['application'])
BATCH_SIZE = int(dict['batch_size'])
LEARNING_RATE = float(dict['learning_rate'])
MAX_STEPS = int(dict['max_steps'])
DISPLAY_STEPS = int(dict['display_steps'])
OPTIMIZER = (dict['optimizer'])
TV_WEIGHT = float(dict['tv_weight'])
REF_IDX = float(dict['ref_idx'])
LAMDA = float(dict['lamda'])
OBJECT_MASK_DISTANCE = float(dict['object_mask_distance'])
MASK_MASK_DISTANCE = float(dict['mask_mask_distance'])
MASK_SENSOR_DISTANCE = float(dict['mask_sensor_distance'])
TRAINING_DATA_TYPE = (dict['training_data_type'])
TESTING_DATA_TYPE = (dict['testing_data_type'])
CAPTURED_DATA_PATH = (dict['captured_data_path'])
OBJECT_PHASE_INPUT = bool(int(dict['object_phase_input']))
OBJECT_AMPLITUDE_INPUT = bool(int(dict['object_amplitude_input']))
OBJECT_ROW = int(dict['object_row'])
OBJECT_COL = int(dict['object_col'])
OBJECT_UPSAMPLING = int(dict['object_upsampling'])
OBJECT_PIXEL_SIZE = float(dict['object_pixel_size'])
MASK_PHASE_MODULATION = bool(int(dict['mask_phase_modulation']))
MASK_AMPLITUDE_MODULATION = bool(int(dict['mask_amplitude_modulation']))
MASK_INIT_TYPE = (dict['mask_init_type'])
MASK_INIT_VALUE = float(dict['mask_init_value'])
MASK_INIT_PATH = (dict['mask_init_path'])
MASK_SAVING_PATH = (dict['mask_saving_path'])
MASK_TESTING_PATH = (dict['mask_testing_path'])
MASK_NUMBER = int(dict['mask_number'])
MASK_ROW = int(dict['mask_row'])
MASK_COL = int(dict['mask_col'])
MASK_PIXEL_SIZE = float(dict['mask_pixel_size'])
SENSOR_ROW = int(dict['sensor_row'])
SENSOR_COL = int(dict['sensor_col'])
SENSOR_PIXEL_SIZE = float(dict['sensor_pixel_size'])
SENSOR_SAVING_PATH = (dict['sensor_saving_path'])
TENSORBOARD_PATH = (dict['tensorboard_path'])
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def import_parameters(file_path):
    dict = {}
    for line in open(file_path):
        tmp = line.split(' ')
        dict[tmp[0]] = tmp[2][:-1]
    return dict


def object_read(path):
    data_list = os.listdir("./" + path)
    data_list_len = len(data_list)
    data_first = np.array(Image.open("./" + path + "/" + data_list[0])) / 255.0
    data_row, data_col = data_first.shape
    data = np.zeros((data_list_len, data_row, data_col))

    for i in range(data_list_len):
        data[i, :, :] = np.array(Image.open("./" + path + "/" + data_list[i])) / 255.0

    data = np.reshape(data, (data_list_len, data_row * data_col))

    return data
