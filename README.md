# Mask-Detector-CNN-python
Mask detector CNN model with 3 layers of convolution

One preprocessing python script

training neural network script

predicting script taking input from frames through webcam live

CNN MODEL
from 100x100x3 image to grayscale image
INPUT IMAGE = 100x100x1
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 98, 98, 50)        500
_________________________________________________________________
bn0 (BatchNormalization)     (None, 98, 98, 50)        200
_________________________________________________________________
activation_1 (Activation)    (None, 98, 98, 50)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 49, 49, 50)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 47, 47, 100)       45100
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 23, 23, 100)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 21, 21, 200)       180200
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 10, 10, 200)       0
_________________________________________________________________
flatten_1 (Flatten)          (None, 20000)             0
_________________________________________________________________
dropout_1 (Dropout)          (None, 20000)             0
_________________________________________________________________
dense_1 (Dense)              (None, 50)                1000050
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 102
=================================================================
Total params: 1,226,152
Trainable params: 1,226,052
Non-trainable params: 100
