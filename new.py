import numpy as np
import csv
from random import random

class Perceptron:

    def __init__(self, inputs, nOut):
        self.inputs = inputs
        self.weights = np.random.rand(nOut)
        self.bias = np.random.rand()
        self.output = 0.0
        self.delta = 0.0
        self.db = 0.0

class MLP:

    def __init__(self, layers):
        self.layers = layers

    def forwardProp(self):
        output = []
        for i, layer in enumerate(self.layers):
            new_inputs = output
            output = []
            print("I:",i)
            if i!=0:
                #get weights of previous layer's perceptrons
                prev_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
                for j, perceptron in enumerate(layer):
                    if new_inputs:
                        perceptron.inputs = new_inputs
                    incoming_weights = []
                    for ar in prev_weights:
                        incoming_weights.append(ar[j])
                    #weights . inputs + bias 
                    if(i==len(self.layers)-1):
                        print("Bias:",perceptron.bias)
                    s = sum(np.multiply(perceptron.inputs, incoming_weights)) + perceptron.bias
                    new_activation = self.sigmoid(s)
                    perceptron.output = new_activation
                    output.append(new_activation)
                    #print("Output:",output)
                
        #print(output)
        return output

    def backwardProp(self, correct):
        outputs = []
        for i, layer in reversed(list(enumerate(self.layers))):
            for perceptron in layer:
                f_prime = perceptron.output*(1.0-perceptron.output) #sigmoid derivative
                delta = (correct-perceptron.output)*f_prime
                outputs.append(delta)
                perceptron.delta = delta

        return outputs

    def gradientDescent(self, p):
        for i in range(len(self.layers)):
            #print("i:",i)
            if i<len(self.layers)-1:
                deltas = []
                for j in range(len(self.layers[i][0].weights)):
                    deltas.append(self.layers[i+1][j].delta) #get deltas of next layer's perceptrons
                #print("Deltas:",deltas)
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
        for epoch in range(epochs):   # repeat for N epochs
            for i, row in enumerate(dataset):  # for every row
                self.forwardProp()
                self.backwardProp(1)   # correct value = last column in input data
                self.gradientDescent(p)
                if i != len(dataset)-1:
                    new_inputs = dataset[i+1]
                # print("new inputs:", new_inputs)
                for j, perceptron in enumerate(self.layers[0]):
                    perceptron.inputs = [new_inputs[j]]
                    # print("neuron inputs:", neuron.inputs)
            print("epochs:", str(epoch+1))
        
if __name__ == "__main__":
    file = open('test_data.csv', 'r')
    data_reader = csv.reader(file, delimiter=',')
    dataset_str = [row[1:9] for row in data_reader]   # only number values from table
    dataset_numstr = dataset_str[2:98]   # remove 98 when passing cleaned data
    dataset = [[float(dataset_numstr[i][j]) for j in range(len(dataset_numstr[i]))]   # convert all str columns to float
           for i in range(len(dataset_numstr))]

    nHidden = int(input("Enter the number of hidden neurons:")) #get number of hidden nodes
    
    input_layer = [Perceptron([column], nHidden) for column in dataset[0]] #initialise input layer
    hidden_layer = [Perceptron([0]*len(input_layer), 1) for k in range(nHidden)] #initialise hidden layer
    output_layer = [Perceptron([0]*len(hidden_layer), 1)] #initialise output layer
    
    # create network
    mlp = MLP([input_layer, hidden_layer, output_layer])
    #train
    mlp.train(dataset,1000,0.5)
    # make a prediction:
    output = mlp.forwardProp()[-1]    # get output from output layer
    print("Next predicted mean daily flow at Skelton:", Calculator.destandardise(output))
