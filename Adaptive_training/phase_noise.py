import numpy as np
import scipy.io as sio
MASK_PIXEL_NUM = 600 * 600
phase_noise = np.random.normal(0, 0.26 * np.pi, (1, MASK_PIXEL_NUM))
sio.savemat('./onn_output/mask_phase_noise_0.26pi.mat', {'phase_noise':phase_noise})

