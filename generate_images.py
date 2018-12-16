# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.utils import np_utils

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
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
import shutil

#--------------------------GLLOBAL VARIABLES------------------------------
FRAME_SIZE = 180

#-------------------------------------------------------------------------
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

def set_class(x):
    if x == 'G1':
        return 1
    elif ('S' in x and 'G1' in x):
        return 1
    elif x == 'S':
        return 2
    elif ('S' in x and 'G2' in x):
        return 2
    elif x == 'G2':
        return 2
    else:
        return 0

def generate_images():
    count_files = 0
    count_cells = 0
    imag_count = 0
    dir = '../Original_Data/'
    dirs = []
    dirs.append(dir)

    for dir in dirs:
        print("DIR: ", dir)
        cells = []

        i = 0  # Used to transverse the cells list when printing the images
        for roots, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.mat'):
                    path = os.path.realpath(os.path.join(roots, file))
                    print("PATH: ", path)
                    data = (sio.loadmat(path, struct_as_record=True))['storage']

                    for case in data:
                        count_cells += 1
                        if (set_class(case['CellCycle'][0][0]) < 3):
                            cells.append(Cell_Info(np.matrix(case['Mask'][0]), case['CellCycle'][0][0]))

                    count_files += 1

        print(count_files, "file(s) found")
        print(count_cells, "cell(s) found,", len(cells), "cell(s) used")

    treated_cells = []

    for cell in cells:
        S_mask = np.zeros((FRAME_SIZE, FRAME_SIZE))

        y_diff = cell.y_max - cell.y_min
        x_diff = cell.x_max - cell.x_min

        if (y_diff > FRAME_SIZE or x_diff > FRAME_SIZE):
            print("Impossible to fit cell, please increase frame size")
        else:
            y_offset = int((FRAME_SIZE - y_diff) / 2)
            x_offset = int((FRAME_SIZE - x_diff) / 2)

            S_mask[y_offset:y_diff + y_offset + 1, x_offset:x_diff + x_offset +
                   1] = cell.Matrix[cell.y_min: cell.y_max + 1, cell.x_min:cell.x_max + 1]
            treated_cells.append(Cell_Info(S_mask.astype(float), cell.CellCycle))

    del cells

    data_G1 = np.array([(cell.Matrix) for cell in treated_cells if (cell.Class == 1)])
    data_G2 = np.array([(cell.Matrix) for cell in treated_cells if cell.Class == 2])


    data_G1_train, data_G1_test = model_selection.train_test_split(data_G1, shuffle=True, test_size=0.10)
    data_G2_train, data_G2_test = model_selection.train_test_split(data_G2, shuffle=True, test_size=0.10)

    data_G1_test, data_G1_validate = model_selection.train_test_split(data_G1_test, shuffle=True, test_size=0.10)
    data_G2_test, data_G2_validate = model_selection.train_test_split(data_G2_test, shuffle=True, test_size=0.10)

    data_list=[data_G1_train, data_G1_test, data_G1_validate,
               data_G2_train, data_G2_test, data_G2_validate]

    directories_list=["../Treated_Data/Images/train/G1", "../Treated_Data/Images/test/G1", "../Treated_Data/Images/validate/G1",
                      "../Treated_Data/Images/train/G2", "../Treated_Data/Images/test/G2", "../Treated_Data/Images/validate/G2"]


    for pos_list in range(len(directories_list)):
        if os.path.exists(directories_list[pos_list]):
            shutil.rmtree(directories_list[pos_list], ignore_errors=True)

        if not os.path.exists(directories_list[pos_list]):
            os.makedirs(directories_list[pos_list])

        for image in data_list[pos_list]:
            filename=directories_list[pos_list]+"/image_"+str(imag_count)+".png"
            cv2.imwrite(filename, image)
            imag_count += 1

    total_cells=len(data_G1)+len(data_G2)
    print(total_cells)
    filename=directories_list[0]+"/image_0.png"
    print(cv2.imread(filename,0).sum(), data_G1_train[0].sum())

generate_images()
