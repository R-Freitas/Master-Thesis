# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Input
from keras.models import Model

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
import h5py

#MODELS
from keras.applications.vgg16 import VGG16

#Paths to save network weights and to obtain info
image_directory = '../Treated_Data/Images'
#weights_load_path = '../Results/Current_Training/best-04-0.84.hdf5'
weights_load_path = None
weights_save_path = '../Results/Current_Training'
history_save_directory = '../Results/History_Objects'
history_name = '/tester'
#Image generator settings
train_batch_size = 10
train_images_generated = 50000
validate_batch_size = 10

#Essential Network settings
epochs = 2
learning_rate = 0.01
model_num = 2 #1 for own model/ 2 for VGG


def create_model(weights_path=None):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(180, 180,1)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation=tf.nn.relu, use_bias=True))
    #model.add(keras.layers.Dropout(0.35))
    model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer=keras.optimizers.Adam(lr = learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(180, 180,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation=tf.nn.sigmoid))

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    return model

def pretrained_VGG(weights_path=None):
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=(180,180,1))

    #Create your own input format (here 3x200x200)
    input = Input(shape=(1,180,180),name = 'image_input')

    #Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation=tf.nn.sigmoid, name='predictions')(x)

    #Create your own model
    model = Model(input=input, output=x)
    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def test_channel_pos():
    if K.image_data_format() == 'channels_first':
        print("FIRST")
        exit()
    else:
        print("LAST")
        exit()
print("RUNNING CONVULOTIONAL NETWORK")
#CNN_Network = create_model(weights_load_path)

if model_num == 2 :
    CNN_Network = VGG_16(weights_load_path)

elif model_num == 1:
    CNN_Network = create_model(weights_load_path)
else:
    print("Select valid model")
    exit()

train_datagen = ImageDataGenerator(
                rescale = 1./255.,
                horizontal_flip = True,
                vertical_flip = True,
                height_shift_range = int(40),
                width_shift_range = int(40),
                rotation_range = 359,               #causa variaçoes reduzidas
                fill_mode = 'constant',
                cval = 0,
                data_format='channels_last')

test_datagen = ImageDataGenerator(
               rescale = 1./255.,
               data_format='channels_last')

train_generator = train_datagen.flow_from_directory(
        image_directory + '/train',  # this is the target directory
        target_size = (180, 180),  # all images will be resized to 150x150
        batch_size = train_batch_size,
        color_mode = 'grayscale',
        class_mode = 'binary',
        shuffle = 'True')

validation_generator = test_datagen.flow_from_directory(
        image_directory + '/test',  # this is the target directory
        target_size = (180, 180),
        batch_size = validate_batch_size,
        color_mode = 'grayscale',
        class_mode = 'binary',
        shuffle = 'True')

cross_validation_generator = test_datagen.flow_from_directory(
        image_directory + '/validate',  # this is the target directory
        target_size = (180, 180),
        batch_size = validate_batch_size,
        color_mode = 'grayscale',
        class_mode = 'binary',
        shuffle = 'True')



TESTING_MODE=0
if TESTING_MODE == 0 :
    filepath= weights_save_path + '/' + str(model_num) + '_{epoch:02d}-{val_acc:.2f}.h5'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, monitor='val_acc', mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)

    try:
        hist=CNN_Network.fit_generator(
                train_generator,
                steps_per_epoch = train_images_generated // train_generator.batch_size,
                epochs = epochs,
                validation_data = validation_generator,
                validation_steps = validation_generator.n // validation_generator.batch_size,
                shuffle = True,
                callbacks=[checkpointer,earlystopping])

        hist = hist.history
        with open(history_save_directory + history_name + '.pkl', 'wb') as f:
                pickle.dump(hist, f)

        acc = hist['acc']
        val_acc = hist['val_acc']
        loss = hist['loss']
        val_loss = hist['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_acc, 'ro', label='Validation accuracy')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')

        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    except KeyboardInterrupt:
        print("Program interrupted, attempting to save.")
        CNN_Network.save_weights(weights_save_path + '/interrupted.h5')
        print('Output saved to: "{}./*"'.format(weights_save_path))



else:
    print("Evaluation results:")
    print(CNN_Network.evaluate_generator(cross_validation_generator, steps=len(cross_validation_generator), max_queue_size=10, workers=1, use_multiprocessing=False))
