import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pickle
import os
from sklearn import preprocessing
import scipy.io as sio
import matplotlib.gridspec as gridspec
import progressbar
from pprint import pprint
import copy




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


class Cell:
    def __init__(self, totalintensity, area, cellcycle):
        self.totalintensity = int(totalintensity)
        self.area = int(area)
        self.cellcycle = str(cellcycle)
        self.Class = set_class(str(cellcycle))



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
        cells=[]
        count=0
        dir = os.getcwd()
        for roots, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.mat'):
                    path = os.path.realpath(os.path.join(roots,file))
                    print(path)
                    data = (sio.loadmat(path,struct_as_record=True))['storage']
                    for case in data:
                        cells.append(Cell(case['TotalIntensity'], case['Area'], case['CellCycle'][0][0]))
                    count += 1

        print (count," files found")


        labeled_data=np.empty((0,3), int)
        for cell in cells:
            labeled_data=np.vstack([labeled_data,[cell.totalintensity,cell.area,cell.Class]])

        np.random.shuffle(labeled_data)

        """
        fac_int = np.amax(labeled_data[:,0])+0.01
        #fac_area=1
        fac_area = np.amax(labeled_data[:,1])+0.01
        scaled_data = np.asfarray(np.column_stack((labeled_data[:,0]/fac_int, labeled_data[:,1]/fac_area,labeled_data[:,2])))
        """

        scaled_data=np.empty((len(labeled_data),3),float)
        scaled_data[:,0]=preprocessing.scale(labeled_data[:,0])
        scaled_data[:,1]=preprocessing.scale(labeled_data[:,1])
        scaled_data[:,2]=labeled_data[:,2]



        size_of_learn_sample = int(len(scaled_data)*0.9)
        train_data = scaled_data[:size_of_learn_sample]
        test_data = scaled_data[-size_of_learn_sample:]

        test_labels = np.array(test_data[:,2]).astype(int)
        test_data = np.asfarray(np.column_stack((test_data[:,0],test_data[:,1])))

        train_labels = np.array(train_data[:,2]).astype(int)
        train_data = np.asfarray(np.column_stack((train_data[:,0],train_data[:,1])))

        no_of_different_labels = 3
        train_labels_one_hot   = np.zeros((train_labels.size,no_of_different_labels))
        test_labels_one_hot    = np.zeros((test_labels.size,no_of_different_labels))

        train_labels_one_hot[np.arange(train_labels.size),train_labels] = 1
        test_labels_one_hot[np.arange(test_labels.size),test_labels]    = 1

        train_labels_one_hot[train_labels_one_hot==0] = 0.01
        train_labels_one_hot[train_labels_one_hot==1] = 0.99
        test_labels_one_hot[test_labels_one_hot==0] = 0.01
        test_labels_one_hot[test_labels_one_hot==1] = 0.99

        #train_data=preprocessing.scale(train_data)
        #test_data=preprocessing.scale(test_data)

        #print(np.amax(scaled_data[:,0]),np.amax(scaled_data[:,1]))
        #print(np.amin(scaled_data[:,0]),np.amin(scaled_data[:,1]))
        print(len(scaled_data))

        fig, sub = plt.subplots(3, 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        gs = gridspec.GridSpec(3, 1)


        ax1 = plt.subplot(gs[0, :])
        cf = ax1.scatter(labeled_data[:,0],labeled_data[:,1],c=labeled_data[:,2], cmap=plt.cm.get_cmap('coolwarm',3), s=10, edgecolors='k')
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Area")
        ax1.set_title("No Change")

        ax1 = plt.subplot(gs[1, :])
        cf = ax1.scatter(scaled_data[:,0],scaled_data[:,1],c=scaled_data[:,2], cmap=plt.cm.get_cmap('coolwarm',3), s=10, edgecolors='k')
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Area")
        ax1.set_title("Change")

        ax1 = plt.subplot(gs[2, :])
        cf = ax1.scatter(train_data[:,0],train_data[:,1],c=train_labels[:], cmap=plt.cm.get_cmap('coolwarm',3), s=10, edgecolors='k')
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Area")
        ax1.set_title("Change")

        #plt.show()



        with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "bw") as fh:
            data = (train_data,
                    test_data,
                    train_labels,
                    test_labels,
                    train_labels_one_hot,
                    test_labels_one_hot)
            pickle.dump(data, fh)


    else:

        with open("/Users/Rafa/Google Drive/Faculdade/Tese/Projecto/Treated_Data/pickled_cells.pkl", "br") as fh:
            data = pickle.load(fh)


        train_data = data[0]
        test_data = data[1]
        train_labels = data[2]
        test_labels = data[3]
        train_labels_one_hot = data[4]
        test_labels_one_hot = data[5]






        epochs = 30000
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
