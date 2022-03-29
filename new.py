import numpy as np
import csv
from random import random
import matplotlib.pyplot as plt

np.random.seed()

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
                    #print("Perceptron Inputs",perceptron.inputs,"j:",j," i:",i)
                    s = sum(np.multiply(perceptron.inputs, incoming_weights)) + perceptron.bias #weights . inputs + bias 
                    new_activation = self.sigmoid(s) #apply the sigmoid function
                    perceptron.output = new_activation
                    output.append(new_activation)
            else:
                for j,perceptron in enumerate(layer):
                    output.append(perceptron.inputs[0])
        return new_activation

    def backwardProp(self, correct):
        error=0.0
        for i, layer in reversed(list(enumerate(self.layers))): #loop through each layer in the MLP in reverse
            for perceptron in layer: #loop through each perceptron in the current layer
                f_prime = perceptron.output*(1.0-perceptron.output) #sigmoid derivative
                delta = (correct-perceptron.output)*f_prime #calculate delta
                perceptron.delta = delta
                if i==2:
                    error = correct-perceptron.output
                    error = error*error
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
        rmse_errors=[]
        for epoch in range(epochs):   #loop for given number of epochs
            errors=[]
            for i, row in enumerate(dataset):  #loop through every row in the data set
                correct = row[4] #define correct value as the value for Skelton on the proceeding day
                prediction = self.forwardProp() #perform forward propogation of the network
                error = self.backwardProp(correct)  #perform backward propogation on the network
                errors.append(error)
                if i==len(dataset)-1:
                    rmse = self.rmse(errors)
                    rmse_errors.append(rmse)
                self.gradientDescent(p) #update weights and biases
                if i != len(dataset)-1:
                    new_inputs = dataset[i+1]
                for j, perceptron in enumerate(self.layers[0]): #define inputs for each 
                    perceptron.inputs = [new_inputs[j]]
        return rmse_errors

    def predict(self, dataset):
        predictions=[]
        correct_values=[]
        for i, row in enumerate(dataset):
            correct_values.append(row[4])
            predictions.append(self.forwardProp())
            if i != len(dataset)-1:
                new_inputs = dataset[i+1]
            for j, perceptron in enumerate(self.layers[0]): #define inputs for each
                perceptron.inputs = [new_inputs[j]]
        plt.plot(predictions)
        plt.plot(correct_values)
        plt.legend(["Predictions","Correct Values"])
        plt.title("Actual vs Modelled Values for Skelton Mean Daily Flow")
        plt.show()
        
    def rmse(self, errors):
        return np.sqrt(sum(errors)/len(errors))    
        
if __name__ == "__main__":
    # train data
    train_file = open('train.csv', 'r')
    train_reader = csv.reader(train_file, delimiter=',')
    train_str = [row[0:5] for row in train_reader]  # only number values from table
    train_data = [[float(train_str[i][j]) for j in range(len(train_str[i]))]  # convert all str columns to float
                  for i in range(len(train_str))]

    # test data
    test_file = open('test.csv', 'r')
    test_reader = csv.reader(test_file, delimiter=',')
    test_str = [row[0:5] for row in test_reader]  # only number values from table
    test_data = [[float(test_str[i][j]) for j in range(len(test_str[i]))]  # convert all str columns to float
                 for i in range(len(test_str))]

    # validation data
    validation_file = open('validation.csv', 'r')
    validation_reader = csv.reader(validation_file, delimiter=',')
    validation_str = [row[0:5] for row in validation_reader]  # only number values from table
    validation_data = [[float(validation_str[i][j]) for j in range(len(validation_str[i]))]  # convert all str columns to float
                       for i in range(len(validation_str))]

    nHidden = int(input("Enter the number of hidden neurons:")) #get number of hidden nodes
    
    input_layer = [Perceptron([train_data[0][i]], nHidden) for i in range(len(train_data[0])-1)] #initialise input layer
    hidden_layer = [Perceptron([0]*len(input_layer), 1) for k in range(nHidden)] #initialise hidden layer
    output_layer = [Perceptron([0]*len(hidden_layer), 1)] #initialise output layer
    
    # create network
    mlp = MLP([input_layer, hidden_layer, output_layer])
    epochs=10
    p=0.01
    #train
    train_errors = mlp.train(train_data,epochs,p)
    test_errors = mlp.train(test_data,epochs,p)
    x = [i for i in range(epochs)]
    plt.plot(x,train_errors)
    plt.title("RMSE Errors for Training Data")
    plt.xlabel("Number of Epochs (n)")
    plt.ylabel("RMSE Error")
    plt.show()

    plt.plot(x,test_errors)
    plt.title("RMSE Errors for Test Data")
    plt.xlabel("Number of Epochs (n)")
    plt.ylabel("RMSE Error")
    plt.show()
    
    mlp.predict(validation_data)
