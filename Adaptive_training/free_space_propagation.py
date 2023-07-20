import tensorflow as tf

import numpy as np
import math
import time
import sys

def tf_fft_shift_2d(img):
    batch, row, col = img.shape
    if np.mod(row.value, 2) == 0:
        row_middle = row.value / 2.0
    else:
        row_middle = np.floor(row.value / 2.0) + 1.0
    if np.mod(col.value, 2) == 0:
        col_middle = col.value / 2.0
    else:
        col_middle = np.floor(col.value / 2.0) + 1.0

    row_middle = int(row_middle)
    col_middle = int(col_middle)

    img_1 = tf.slice(img, np.int32([0, 0, 0]), np.int32([batch, row_middle, col_middle]))
    img_2 = tf.slice(img, np.int32([0, 0, col_middle]), np.int32([batch, row_middle, col - col_middle]))
    img_3 = tf.slice(img, np.int32([0, row_middle, 0]), np.int32([batch, row - row_middle, col_middle]))
    img_4 = tf.slice(img, np.int32([0, row_middle, col_middle]), np.int32([batch, row - row_middle, col - col_middle]))

    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)


def tf_ifft_shift_2d(img):
    batch, row, col = img.shape
    if np.mod(row.value, 2) == 0:
        row_middle = row.value / 2.0
    else:
        row_middle = np.floor(row.value / 2.0)
    if np.mod(col.value, 2) == 0:
        col_middle = col.value / 2.0
    else:
        col_middle = np.floor(col.value / 2.0)

    row_middle = int(row_middle)
    col_middle = int(col_middle)

    img_1 = tf.slice(img, np.int32([0, 0, 0]), np.int32([batch, row_middle, col_middle]))
    img_2 = tf.slice(img, np.int32([0, 0, col_middle]), np.int32([batch, row_middle, col - col_middle]))
    img_3 = tf.slice(img, np.int32([0, row_middle, 0]), np.int32([batch, row - row_middle, col_middle]))
    img_4 = tf.slice(img, np.int32([0, row_middle, col_middle]), np.int32([batch, row - row_middle, col - col_middle]))

    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)


def BandLimitTransferFunction(pixelsize, z, lamda, Fvv, Fhh):
    hSize, vSize = Fvv.shape
    dU = (hSize * pixelsize) ** -1.0
    dV = (vSize * pixelsize) ** -1.0
    Ulimit = ((2.0 * dU * z) ** 2.0 + 1.0) ** -0.5 / lamda
    Vlimit = ((2.0 * dV * z) ** 2.0 + 1.0) ** -0.5 / lamda
    freqmask = ((Fvv ** 2.0 / (Ulimit ** 2.0) + Fhh ** 2.0 * (lamda ** 2.0)) <= 1.0) & ((Fvv ** 2.0 * (lamda ** 2.0) + Fhh ** 2.0 / (Vlimit ** 2.0)) <= 1.0)
    return freqmask


def PropGeneral(Fhh, Fvv, lamda, refidx, z):
    DiffLimMat = np.ones(Fhh.shape)
    lamdaeff = lamda / refidx
    DiffLimMat[(Fhh ** 2.0 + Fvv ** 2.0) >= (1.0 / lamdaeff ** 2.0)] = 0.0

    temp1 = 2.0 * math.pi * z / lamdaeff
    temp3 = (lamdaeff * Fvv) ** 2.0
    temp4 = (lamdaeff * Fhh) ** 2.0
    temp2 = np.complex128(1.0 - temp3 - temp4) ** 0.5
    H = np.exp(1j * temp1*temp2)
    H[np.logical_not(DiffLimMat)] = 0
    return H


def propagate(img, pixelsize, refidx, lamda, z, convunits=True, tf_zeropad=True, tf_freqmask=True):
    if convunits is True:
        lamda = lamda * 1e-9
        pixelsize = pixelsize * 1e-6
        z = z * 1e-6 #um

    Nv, Nh = img.shape[1], img.shape[2]
    #Haven't Translated the padding
    #if tf_zeropad == True
    #    img = padarray(img, [Nv, Nh], meanvalue, 'post')

    spectrum = tf.fft2d(img)
    spectrum = tf_fft_shift_2d(spectrum)

    batch, NFv, NFh = spectrum.shape

    Fs = 1 / pixelsize
    Fh = Fs / NFh.value * np.arange((-np.ceil((NFh.value - 1) / 2)), np.floor((NFh.value - 1) / 2)+0.5)
    Fv = Fs / NFv.value * np.arange((-np.ceil((NFv.value - 1) / 2)), np.floor((NFv.value - 1) / 2)+0.5)
    [Fhh, Fvv] = np.meshgrid(Fh, Fv)

    np_H = PropGeneral(Fhh, Fvv, lamda, refidx, z)
    np_freqmask = BandLimitTransferFunction(pixelsize, z, lamda, Fvv, Fhh)
    H = tf.convert_to_tensor(np_H, dtype=tf.complex64)
    freqmask = tf.convert_to_tensor(np_freqmask)
    spectrum_z = tf.multiply(spectrum, H)

    if tf_freqmask is True:
        spectrum_z = tf.multiply(spectrum_z, tf.cast(freqmask, tf.complex64))

    spectrum_z = tf_ifft_shift_2d(spectrum_z)
    img_z = tf.ifft2d(spectrum_z)
    img_z = tf.slice(img_z, np.int32([0, 0, 0]), np.int32([batch, Nv.value, Nh.value]))

    return img_z


def batch_propagate(img, batch_shape, pixelsize, refidx, lamda, z, convunits, tf_zeropad, tf_freqmask):
    # start_time = time.clock()
    img_reshaped = tf.reshape(img, batch_shape)

    img_prop = propagate(img_reshaped, pixelsize, refidx, lamda, z, convunits, tf_zeropad, tf_freqmask) # img_reshaped is 3D
    img_prop_reshaped = tf.reshape(img_prop, [batch_shape[0], batch_shape[1]*batch_shape[2]])
    # sys.stdout.write('\nbatch propagate Time %f' % (time.clock()-start_time))
    # sys.stdout.flush()
    return img_prop_reshaped
