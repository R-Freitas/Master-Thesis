import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import truncnorm
import pickle
import os
from sklearn import preprocessing
import scipy.io as sio
import matplotlib.gridspec as gridspec
import progressbar
from pprint import pprint
import copy


input_size = 140

def truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


activation_function = sigmoid


class NeuralNetwork:


    def __init__(self,
                 network_structure, # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
                 learning_rate,
                 bias=None
                ):
        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()


    def create_weight_matrices(self):
        X = truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)

        bias_node = 1 if self.bias else 0

        self.weights_matrices = []
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1


    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            # adding bias node to the end of the 'input'_vector
            if self.bias:
                in_vector = np.concatenate((in_vector,[[self.bias]]))
                res_vectors[-1] = in_vector

            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = activation_function(x)
            res_vectors.append(out_vector)
            layer_index += 1

        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
         # The input vectors to the various layers
        output_errors = target_vector - out_vector

        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]
            if self.bias and not layer_index==(no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
            tmp = output_errors * out_vector * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)

            self.weights_matrices[layer_index-1] += self.learning_rate * tmp

            output_errors = np.dot(self.weights_matrices[layer_index-1].T,output_errors)
            if self.bias:
                output_errors = output_errors[:-1,:]
            layer_index -= 1


    def train(self, data_array,
              labels_one_hot_array,
              epochs=1,
              intermediate_results=False):

        bar = progressbar.ProgressBar(maxval=epochs, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        intermediate_weights = []
        count=1
        for epoch in range(epochs):
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append(copy.deepcopy(self.weights_matrices))
            bar.update(count)
            count += 1

        bar.finish()

        return intermediate_weights



    def run(self, input_vector):

        # input_vector can be tuple, list or ndarray
        no_of_layers = len(self.structure)
        # adding bias node to the end of the inpuy_vector
        if self.bias:
            input_vector = np.concatenate( (input_vector, [self.bias]) )
        in_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index-1],in_vector)
            out_vector = activation_function(x)
            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate( (in_vector,[[self.bias]]) )
            layer_index += 1


        return out_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

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
        return 0

def set_color(x):
    if x == 'G1':
        return 'red'
    elif ('S' in x and 'G1' in x):
        return 'yellow'
    elif x == 'S':
        return 'blue'
    elif ('S' in x and 'G2' in x):
            return 'orange'
    elif x == 'G2':
        return 'green'
    else:
        return 'black'




class Mask:
    def __init__(self,Matrix,Index,CellCycle):
        self.Matrix = Matrix
        self.Index = Index
        self.y_min = min(self.Index[:,0])
        self.y_max = max(self.Index[:,0])
        self.x_min = min(self.Index[:,1])
        self.x_max = max(self.Index[:,1])
        self.y_diff = self.y_max-self.y_min
        self.x_diff = self.x_max-self.x_min
        self.Area = np.count_nonzero(self.Matrix)
        self.Intensity = self.Matrix.sum()
        self.CellCycle = str(CellCycle)
        self.Class = set_class(str(CellCycle))


#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
#____________________________________MAIN_______________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________

if __name__ == "__main__":

    new_data=1
    if (new_data==1) :
        count=0
        dir = os.getcwd()
        dirs=[]
        dirs.append(dir)




        for dir in dirs:
            print("DIR: ",dir)
            cells=[]
            masks=[]
            i=0 #Used to transverse the masks list when printing the images
            for roots, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith('.mat'):
                        path = os.path.realpath(os.path.join(roots,file))
                        print("PATH: ",path)
                        data = (sio.loadmat(path,struct_as_record=True))['storage']

                        for case in data:
                            matrix=np.matrix(case['Mask'][0])
                            index=np.asarray(np.where(matrix>0)).T
                            masks.append(Mask(matrix,index,case['CellCycle'][0][0]))

                        count += 1

                        """
                        #Routine used to print all cells from a mat file as an image
                        fig=plt.figure(frameon=False)
                        final_mask=np.zeros_like(masks[0].Matrix)
                        for index in range(i,len(masks)):
                            final_mask += masks[index].Matrix
                            i += 1

                        plt.imshow(final_mask, cmap='Blues',interpolation='nearest')
                        plt.show()
                        """

            print(count, "files found")
            print(len(masks), "cells found")


            """
            #Routine used to determine the maximum cell size and thus choose an
            #appropriate input size (in this case 140)
            pix_size=[]
            for mask in masks:
                pix_size.append([mask.y_diff,mask.x_diff])

            pix_size=np.array(pix_size)
            print(np.amax(pix_size,axis=0))
            """


            """
            #Routin used to check all information is correct
            print('=================================================')
            for i in range(10):
                print(masks[i].y_min,masks[i].y_max)#Y min and max
                print(masks[i].x_min,masks[i].x_max)#X min and max
                print(masks[i].Intensity)
                print(masks[i].Area)
                print(masks[i].CellCycle)
                print(masks[i].Class)
                print('=================================================')
            """

            #With all the cells masks in a list, and an input size chosen it is
            #time to create the input for the neural network itself





        """
        with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "bw") as fh:
            data = (train_data,
                    train_labels,
                    test_labels,
                    train_labels_one_hot,
                    test_labels_one_hot)
            pickle.dump(data, fh)
        """

    else:

        with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "br") as fh:
            data = pickle.load(fh)


        train_data = data[0]
        test_data = data[1]
        train_labels = data[2]
        test_labels = data[3]
        train_labels_one_hot = data[4]
        test_labels_one_hot = data[5]






        epochs = 10
        test_epochs=[500,1000,5000,10000,15000,20000,25000,30000]

        ANN = NeuralNetwork(network_structure=[2, 150, 150, 3],
                                   learning_rate=0.01,
                                   bias=1)

        print("Epochs: ",epochs, "\tTraining Size: ",len(train_data),"\tStructure: ",ANN.structure,"\tBias: ",ANN.bias,"\tLearning Rate: ",ANN.learning_rate)
        matrices=ANN.train(train_data, train_labels_one_hot, epochs=epochs, intermediate_results=True)
        i=1
        """
        print("============================================================================================")
        corrects, wrongs = ANN.evaluate(train_data, train_labels)
        print("accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = ANN.evaluate(test_data, test_labels)
        print("accuracy test: ", corrects / ( corrects + wrongs))
        """
        print("============================================================================================")

        for element in matrices:
            if (i in test_epochs):
                ANN.weights_matrices = element
                print("Epochs: ",i)
                corrects, wrongs = ANN.evaluate(train_data, train_labels)
                print("accuracy train: ", corrects / ( corrects + wrongs))
                corrects, wrongs = ANN.evaluate(test_data, test_labels)
                print("accuracy: test", corrects / ( corrects + wrongs))
                print("============================================================================================")
            i += 1
