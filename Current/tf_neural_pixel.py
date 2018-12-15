
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

#--------------------------GLLOBAL VARIABLES------------------------------
FRAME_SIZE = 140

#-------------------------------------------------------------------------


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
    #data = np.array([(cell.Matrix) for cell in treated_cells])
    #labels = to_categorical(np.array([(int(cell.Class)) for cell in treated_cells]),num_classes=3)
    data_G1=np.array([(cell.Matrix) for cell in treated_cells if cell.Class==0])
    data_S=np.array([(cell.Matrix) for cell in treated_cells if cell.Class==1])
    data_G2=np.array([(cell.Matrix) for cell in treated_cells if cell.Class==2])

    labels_G1=np.empty(len(data_G1))
    labels_G1.fill(0)
    labels_G1 = to_categorical(labels_G1,num_classes=3)

    labels_S=np.empty(len(data_S))
    labels_S.fill(1)
    labels_S = to_categorical(labels_S,num_classes=3)

    labels_G2=np.empty(len(data_G2))
    labels_G2.fill(2)
    labels_G2 = to_categorical(labels_G2,num_classes=3)

    labels=np.vstack((labels_G1,labels_S,labels_G2))
    data = np.vstack((data_G1,data_S,data_G2))

    data=data/255.0
    del treated_cells

    print("Data points used: ", len(data))



    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, shuffle=True, test_size=0.10)
    test_data, validate_data, test_labels, validate_labels = model_selection.train_test_split(test_data, test_labels, shuffle=True, test_size=0.50)

    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells_pixel.pkl", "bw") as fh:
        data = (train_data,
                test_data,
                train_labels,
                test_labels,
                validate_data,
                validate_labels)
        pickle.dump(data, fh)

def load_data():
    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells_pixel.pkl", "br") as fh:
        data = pickle.load(fh)


    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    x_validation = data[4]
    y_validation = data[5]

    return x_train, y_train, x_test, y_test

def load_validation():
    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells_pixel.pkl", "br") as fh:
        data = pickle.load(fh)

    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    x_validation = data[4]
    y_validation = data[5]

    return x_validation, y_validation

def optimize_model(x_train, y_train, x_test, y_test):
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
    model.add(keras.layers.Flatten(input_shape=(140, 140)))

    for i in (range(20)):
        model.add(keras.layers.Dense(1000, activation=tf.nn.relu,use_bias=True))
        model.add(keras.layers.Dropout({{uniform(0,0.5)}}))

    model.add(keras.layers.Dense(3,activation=tf.nn.softmax))




    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=100,
              batch_size=50,
              shuffle=True,
              validation_data=(x_test, y_test),
              verbose=1)


    x_validation, y_validation=load_validation()
    score, acc = model.evaluate(x_validation, y_validation)
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def create_model(x_train, y_train, x_test, y_test):



    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(140, 140)))

    for i in (range(20)):
        model.add(keras.layers.Dense(1000, activation=tf.nn.relu,use_bias=True))
        model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(3,activation=tf.nn.softmax))



    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,mode='auto')]
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history=model.fit(x_train, y_train,
              epochs=100,
              shuffle=True,
              batch_size=50,
              validation_data=(x_test, y_test))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, loss, 'b', label='Training loss')
    # "ro" is for "blue dot"
    plt.plot(epochs, val_acc, 'ro', label='Validation accuracy')
    # r is for "solid blue line"
    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    x_validation, y_validation=load_validation()
    score, acc = model.evaluate(x_validation, y_validation)
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


new_data=0
if (new_data==1) :
    generate_save_data()

else:
    optimizing=0
    if optimizing:
        best_run, best_model = optim.minimize(model=optimize_model,
                                              data=load_data,
                                              algo=tpe.suggest,
                                              max_evals=3,
                                              trials=Trials())

        X_train, Y_train, X_test, Y_test = load_data()
        print("Evalutati on of best performing model:")
        print(best_model.evaluate(X_test,Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)

    else:
        X_train, Y_train, X_test, Y_test = load_data()
        result, status, best_model = create_model(X_train, Y_train, X_test, Y_test)
