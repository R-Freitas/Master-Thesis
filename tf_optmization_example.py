from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import fashion_mnist
from keras.layers.core import Dense, Dropout, Activation
#from keras.models import Sequential
from keras.utils import np_utils



import tensorflow as tf
from tensorflow import keras

from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = keras.Sequential([
        keras.layers.Dense(150, input_shape=(784,), activation=tf.nn.relu,use_bias=True),
        keras.layers.Dense(150, activation=tf.nn.relu,use_bias=True),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])


    model.compile(optimizer={{choice([keras.optimizers.Adam(), 'rmsprop', 'sgd'])}},
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=1,
              batch_size=100,
              shuffle=True,
              validation_data=(x_test, y_test),
              verbose=2)


    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
