# ACCEL (All-analog Chip Combining Electronics and Light)

### Overview
ACCEL is an all-analog chip able to process high-performance image classification and time-lapse tasks. In order to tackle the challenge of sensitivity to the system errors induced by inevitable manufacturing defects and misalignment during packaging, ACCEL adopts adaptive training method to fine-tune the electronic analog computing (EAC) part with back propagation based on the intermediate optical analog computing (OAC) results captured by the photodiode array. The relevant codes are available in ‘./Adaptive_training/’ .

### Environment
We use Python 3.5.2 and Tensorflow 1.14.0. For detailed environment configuration, please refer to './environment.txt'.

### Installation guide
1) The installation of Tensorflow in Python environment can be refenced to the official website of Tensorflow: https://www.tensorflow.org/install/.
2) The dependent packages in the environment.txt file can be installed from PyPi:
```
pip install -r environment.txt
```
### Datasets
The MNIST dataset is included in the path ‘./Adaptive_training/MNIST_data/’, and the video judgement datasets used in the time-lapse task are provided in ‘./video judgment dataset/’. One may also download the standard Fashion-MNIST, K-MNIST and ImageNet datasets to train ACCEL from the following links:
1. Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
2. KMNIST: https://github.com/rois-codh/kmnist
3. ImageNet: https://image-net.org/download-images

### Usage
The code is tested under Python 3.5.2 and Tensorflow 1.14.0 environment with NVIDIA GPU GTX-1080Ti. The ‘./Adaptive_training/’ directory includes the following sub-directories and files:
1. 'ckpt_bwn/': This directory contains the trained checkpoint file of EAC.
2. 'MNIST_data/': This directory contains the MNIST dataset.
3. 'onn_output/': This directory contains three types of disturbed output images of OAC in the case of weight errors from manufacturing defect, pixel shifting and image rotation from misalignment between OAC and EAC. These output images of OAC are used for the adaptive training of ACCEL.
4. 'output_bwn/': This directory contains the log files of testing accuracy during adaptive training, and the files that contains trained weights.
5. 'BWN_train_phase_noise.py', 'BWN_train_shiftRght.py', 'BWN_train_rotate.py': These three files are the python codes for the adaptive training of EAC. Run directly to load the corresponding disturbed OAC output images from ‘onn_output/’ directory and train the weight parameters of EAC.

### License
This project is covered under the GNU General Public License v3.0.
