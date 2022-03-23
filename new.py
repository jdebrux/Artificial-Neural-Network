import numpy as np
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
            print("I:",i)
            if i!=0:
                #get weights of previous layer's perceptrons
                prev_weights = [self.layers[i-1][j].weights for j in range(len(self.layers[i-1]))]
                for j, perceptron in enumerate(layer):
                    incoming_weights = []
                    for ar in prev_weights:
                        incoming_weights.append(ar[j])
                    #weights . inputs + bias 
                    #print("IN",perceptron.inputs)
                    s = sum(np.multiply(perceptron.inputs, incoming_weights)) + perceptron.bias
                    new_activation = self.sigmoid(s)
                    perceptron.output = new_activation
                    output.append(new_activation)
                    print("Output:",new_activation)
                
        #print(output)
        return output

    def backwardProp(self):
        correct=1
        outputs = []
        for i, layer in reversed(list(enumerate(self.layers))):
##            activations=[]
##            for perceptron in self.layers[i-1]: #get previous layers outputs
##                activations.append(perceptron.output)
##                delta = error * self.sigmoidDerivative(perceptron.output)
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
                    weight_change = [x*p for x in deltas]
                    weight_change = [x*perceptron.output for x in weight_change]
                    perceptron.weights += weight_change
                    perceptron.bias += p*perceptron.delta #update bias using wi,j = wi,j + p*δj*ui
                    
            
    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    @staticmethod
    def train(self, mlp, dataset, epochs, learning_rate):
        for i in range(epochs):
            
            pass

if __name__ == "__main__":
    n1 = Perceptron([1],3)
    n2 = Perceptron([2],3)
    n3 = Perceptron([3, 3],1)
    n4 = Perceptron([2, 2],1)
    n5 = Perceptron([2, 2],1)
    n6 = Perceptron([2, 2, 2],1)

    l1 = [n1, n2]
    l2 = [n3, n4, n5]
    l3 = [n6]

    mlp = MLP([l1, l2, l3])

    #no target specified - implement and use error in back prop
    epochs = 500
    for i in range(epochs):
        mlp.forwardProp()
        mlp.backwardProp()
        mlp.gradientDescent(0.9)
