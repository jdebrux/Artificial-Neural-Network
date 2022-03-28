import numpy as np
import csv
from random import random

"""
The perceptron class defines the storage of data for a given perceptron in the neural network.
The perceptrons will be used to build up each layer and thus form a Multi Layer Perceptron.
"""
class Perceptron:

    def __init__(self, inputs, nOut):
        self.inputs = inputs
        self.weights = np.random.rand(nOut)
        self.bias = np.random.rand()
        self.output = 0.0
        self.delta = 0.0
        self.db = 0.0
"""
The MLP class represents a Multi-Layer Perceptron, and the methods required to train it.
"""
class MLP:

    def __init__(self, layers):
        self.layers = layers

    def forwardProp(self):
        output = []
        for i, layer in enumerate(self.layers): #loop through each layer in the MLP
            #store the inputs for a given layer as the outputs of the previous layer
            new_inputs = output 
            output = []
            if i!=0: #check if we are at the input layer
                #get weights of previous layer's perceptrons
                prev_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
                for j, perceptron in enumerate(layer): #loop through each perceptron in the layer
                    if new_inputs: #prevent blank inputs
                        perceptron.inputs = new_inputs 
                    incoming_weights = []
                    #loop through each set of previous weights and find the weights correlating to the current perceptron
                    for ar in prev_weights: 
                        incoming_weights.append(ar[j])
                    s = sum(np.multiply(perceptron.inputs, incoming_weights)) + perceptron.bias #weights . inputs + bias 
                    new_activation = self.sigmoid(s) #apply the sigmoid function
                    #if i==1:
                        #print("Old perceptron:", perceptron.output)
                    perceptron.output = new_activation
                    #if i==1:
                        #print("New perceptron:", perceptron.output)
                    output.append(new_activation)
            else:
                for j,perceptron in enumerate(layer):
                    perceptron.output = perceptron.inputs[0]
                
        return new_activation

    def backwardProp(self, correct):
        outputs = []
        error=0.0
        for i, layer in reversed(list(enumerate(self.layers))): #loop through each layer in the MLP in reverse
            for perceptron in layer: #loop through each perceptron in the current layer
                f_prime = perceptron.output*(1.0-perceptron.output) #sigmoid derivative
                delta = (correct-perceptron.output)*f_prime #calculate delta
                perceptron.delta = delta
                #print(i,"|",perceptron.output)
                #print("ERROR",correct-perceptron.output)
                if i==2:
                    error = perceptron.output-correct
        return error

    def gradientDescent(self, p):
        for i in range(len(self.layers)):
            #print("i:",i)
            if i<len(self.layers)-1:
                deltas = []
                for j in range(len(self.layers[i][0].weights)):
                    deltas.append(self.layers[i+1][j].delta) #get deltas of next layer's perceptrons
            for perceptron in self.layers[i]:
                if i<len(self.layers)-1:
                    weight_change = [x*p for x in deltas]
                    weight_change = [x*perceptron.output for x in weight_change]
                    perceptron.weights += weight_change
                perceptron.bias += p*perceptron.delta #update bias using wi,j = wi,j + p*δj*ui
                    
            
    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def train(self, dataset, epochs, p):
        for epoch in range(epochs):   #loop for given number of epochs
            for i, row in enumerate(dataset):  #loop through every row in the data set
                correct = row[4] #define correct value as the value for Skelton on the proceeding day
                prediction = self.forwardProp() #perform forward propogation of the network
                error = self.backwardProp(correct)  #perform backward propogation on the network
                if i==len(dataset)-1: #only output for final prediction made
                      print("Prediction:",prediction)
                      print("Error: ",error)
                self.gradientDescent(p) #update weights and biases
                if i != len(dataset)-1:
                    new_inputs = dataset[i+1]
                for j, perceptron in enumerate(self.layers[0]): #define inputs for each 
                    perceptron.inputs = [new_inputs[j]]
            #print("Epoch:", str(epoch+1))
        
if __name__ == "__main__":
    file = open('data.csv', 'r')
    data_reader = csv.reader(file, delimiter=',')
    dataset_str = [row[0:5] for row in data_reader]   # only number values from table
    dataset = [[float(dataset_str[i][j]) for j in range(len(dataset_str[i]))]   # convert all str columns to float
           for i in range(len(dataset_str))]

    nHidden = int(input("Enter the number of hidden neurons:")) #get number of hidden nodes
    
    input_layer = [Perceptron([dataset[0][i]], nHidden) for i in range(len(dataset[0])-1)] #initialise input layer
    hidden_layer = [Perceptron([0]*len(input_layer), 1) for k in range(nHidden)] #initialise hidden layer
    output_layer = [Perceptron([0]*len(hidden_layer), 1)] #initialise output layer
    
    # create network
    mlp = MLP([input_layer, hidden_layer, output_layer])
    #train
    print(len(mlp.layers[0]))
    mlp.train(dataset,1000,0.5)
    # make a prediction:
    output = mlp.forwardProp()   # get output from output layer
    #print("Next predicted mean daily flow at Skelton:", Calculator.destandardise(output))
