
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.utils import np_utils

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


def set_class(x):
    if x == 'G1':
        return 0
    elif ('S' in x and 'G1' in x):
        return 1
    elif x == 'S':
        return 3
    elif ('S' in x and 'G2' in x):
        return 2
    elif x == 'G2':
        return 4
    else:
        return 5

def generate_save_data():
    count_files=0
    count_cells = 0
    dir = os.getcwd()
    dirs=[]
    dirs.append(dir)

    for dir in dirs:
        print("DIR: ",dir)
        cells=[]

        i=0 #Used to transverse the cells list when printing the images
        for roots, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.mat'):
                    path = os.path.realpath(os.path.join(roots,file))
                    print("PATH: ",path)
                    data = (sio.loadmat(path,struct_as_record=True))['storage']

                    for case in data:
                        count_cells += 1
                        if (set_class(case['CellCycle'][0][0]) < 3):
                            cells.append(Cell_Info(np.matrix(case['Mask'][0]),case['CellCycle'][0][0]))

                    count_files += 1

                    """
                    #Routine used to print all cells from a mat file as an image
                    fig=plt.figure(frameon=False)
                    final_mask=np.zeros_like(cells[0].Matrix)
                    for index in range(i,len(cells)):
                        final_mask += cells[index].Matrix
                        i += 1

                    plt.imshow(final_mask, cmap='Blues',interpolation='nearest')
                    plt.show()
                    """

        print(count_files, "file(s) found")
        print(count_cells, "cell(s) found,", len(cells), "cell(s) used")


        """
        #Routine used to determine the maximum cell size and thus choose an
        #appropriate input size (in this case 140x140)
        pix_size=[]
        for cell in cells:
            pix_size.append([(cell.y_max-cell.y_min),(cell.x_max-cell.x_min)])

        pix_size=np.array(pix_size)
        print(np.amax(pix_size,axis=0))
        """

        """
        #Routine used to check if all information is correct
        print('=================================================')
        for i in range(10):
            print("Y:",cells[i].y_min,cells[i].y_max)#Y min and max
            print("X:",cells[i].x_min,cells[i].x_max)#X min and max
            print(cells[i].Intensity)
            print(cells[i].Area)
            print(cells[i].CellCycle)
            print(cells[i].Class)
            print('=================================================')
        """

    #With all the cells cells in a list, and an input size chosen it is
    #time to create the input for the neural network itself
    treated_cells=[]


    for cell in cells:
        S_mask=np.zeros((FRAME_SIZE,FRAME_SIZE))

        y_diff = cell.y_max - cell.y_min
        x_diff = cell.x_max - cell.x_min

        if (y_diff > FRAME_SIZE or x_diff > FRAME_SIZE):
            print("Impossible to fit cell, please increase frame size")
        else:
            y_offset = int((FRAME_SIZE-y_diff)/2)
            x_offset = int((FRAME_SIZE-x_diff)/2)

            S_mask[y_offset:y_diff+y_offset+1,x_offset:x_diff+x_offset+1] = cell.Matrix[cell.y_min : cell.y_max+1, cell.x_min:cell.x_max+1]
            treated_cells.append(Cell_Info(S_mask.astype(float),cell.CellCycle))


    del cells
    data = np.array([(cell.Area,cell.Intensity) for cell in treated_cells])
    labels = to_categorical(np.array([(int(cell.Class)) for cell in treated_cells]),num_classes=3)



    scaler=preprocessing.MinMaxScaler()
    print(np.std(data[:,0]),np.std(data[:,1]))
    data=scaler.fit_transform(data)
    print(np.amax(data[:,0]),np.amax(data[:,1]))
    print(np.std(data[:,0]),np.std(data[:,1]))
    del treated_cells



    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, shuffle=True, test_size=0.10)


    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "bw") as fh:
        data = (train_data,
                test_data,
                train_labels,
                test_labels)
        pickle.dump(data, fh)

def load_data():
    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "br") as fh:
        data = pickle.load(fh)


    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]

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
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=0)]

    model = keras.Sequential()
    model.add(keras.layers.Dense({{choice([500,750,1000])}}, input_shape=(2,), activation=tf.nn.relu,use_bias=True))
    model.add(keras.layers.Dropout({{uniform(0, 1)}}))

    model.add(keras.layers.Dense({{choice([500,750,1000])}}, input_shape=(2,), activation=tf.nn.relu,use_bias=True))
    model.add(keras.layers.Dropout({{uniform(0, 1)}}))

    if ({{choice(['two', 'three'])}}) == 'three':
        model.add(keras.layers.Dense({{choice([500,750,1000])}}, input_shape=(2,), activation=tf.nn.relu,use_bias=True))
        model.add(keras.layers.Dropout({{uniform(0, 1)}}))


    model.add(keras.layers.Dense(3,activation=tf.nn.softmax))



    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=1000,
              batch_size={{choice([32,64,128])}},
              shuffle=True,
              validation_data=(x_test, y_test),
              verbose=0,
              callbacks=callbacks)


    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

class Cell_Info:
    def __init__(self,Matrix,CellCycle):
        self.Matrix = Matrix
        self.Index = np.asarray(np.where(Matrix>0)).T
        self.y_min = min(self.Index[:,0])
        self.y_max = max(self.Index[:,0])
        self.x_min = min(self.Index[:,1])
        self.x_max = max(self.Index[:,1])
        self.Area = np.count_nonzero(self.Matrix)
        self.Intensity = self.Matrix.sum()
        self.CellCycle = str(CellCycle)
        self.Class = set_class(str(CellCycle))

#--------------------------GLLOBAL VARIABLES------------------------------
FRAME_SIZE=140
#-------------------------------------------------------------------------

new_data=0
if (new_data==1) :
    generate_save_data()

else:


    best_run, best_model = optim.minimize(model=create_model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = load_data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test,Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)















    """
    #_________________________FIRST EXAMPLE______________________________________


    class_names = ['G1','S/G1','S/G2']

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)]
    model = keras.Sequential([
        keras.layers.Dense(150, input_shape=(2,), activation=tf.nn.relu,use_bias=True),
        keras.layers.Dense(150, activation=tf.nn.relu,use_bias=True),
        keras.layers.Dense(3,activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=30000,batch_size=100,shuffle=True,validation_data=(test_data, test_labels),callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print('Test accuracy:', test_acc)
    """







#
