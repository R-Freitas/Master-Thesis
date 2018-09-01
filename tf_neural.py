
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

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



    labeled_data = np.array([(cell.Area,cell.Intensity,int(cell.Class)) for cell in treated_cells])
    del cells
    del treated_cells




    #==============DATA RANDOMIZING===============
    np.random.shuffle(labeled_data)
    labeled_data[:,0]=preprocessing.scale(labeled_data[:,0])
    labeled_data[:,1]=preprocessing.scale(labeled_data[:,1])
    size_of_learn_sample = int(len(labeled_data)*0.9)
    train_data = labeled_data[:size_of_learn_sample]
    test_data = labeled_data[size_of_learn_sample:]



    #===============TRAINING DATA=================
    train_labels = to_categorical(train_data[:,2].astype(int),num_classes=3)
    train_data = train_data[:,0:2]



    #===============TESTING DATA===================
    test_labels = to_categorical(test_data[:,2].astype(int),num_classes=3)
    test_data = test_data[:,0:2]





    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "bw") as fh:
        data = (train_data,
                test_data,
                train_labels,
                test_labels)
        pickle.dump(data, fh)


else:

    with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "br") as fh:
        data = pickle.load(fh)


    train_data = data[0]
    test_data = data[1]
    train_labels = data[2]
    test_labels = data[3]




    print(train_data)

    #_________________________FIRST EXAMPLE______________________________________


    class_names = ['G1','S/G1','S/G2']


    model = keras.Sequential([
        keras.layers.Dense(150, input_shape=(2,), activation=tf.nn.relu),
        keras.layers.Dense(150, activation=tf.nn.relu),
        keras.layers.Dense(3,activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10000,batch_size=100,shuffle=True)

    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print('Test accuracy:', test_acc)








#
