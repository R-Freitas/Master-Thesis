# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam

#Optimizar libraries
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.io as sio
import progressbar
import copy
import time
from sklearn import preprocessing
from sklearn import model_selection
import cv2



train_datagen = ImageDataGenerator(
                rescale = 1./255.,
                fill_mode = 'constant',
                cval = 0,
                horizontal_flip = True,
                vertical_flip = True,
                height_shift_range = int(40),
                width_shift_range = int(40),
                rotation_range = 359,               #causa variaçoes reduzidas
                data_format='channels_last')

test_datagen = ImageDataGenerator(
               rescale = 1./255.,
               data_format='channels_last')

train_generator = train_datagen.flow_from_directory(
        '../Treated_Data/Image_test/test',  # this is the target directory
        target_size=(180, 180),  # all images will be resized to 150x150
        batch_size=1,
        color_mode = 'grayscale',
        class_mode= 'binary')


i=0
for inputs,outputs in train_generator:
    print(inputs[0].sum(),np.count_nonzero(inputs[0]))
    plt.imshow(inputs[0].reshape(180,180),cmap=plt.get_cmap('gray'))
    plt.show()
    i += 1
    if i == 10:
        break

filename="../Treated_Data/Image_test/result"+"/image_2300.png"
print(cv2.imread(filename,0).sum()/255,np.count_nonzero(cv2.imread(filename,0)))
