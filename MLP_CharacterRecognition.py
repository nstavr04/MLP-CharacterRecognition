# Implementation of a Feed Forward Neural Network for character recognition
# Author: Nikolas Stavrou

# Inputs: 16 values from 0-15 representing the vector values of the character
# Output: a 26 bit encoding of the character

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd

class NeuralNetwork_MultiLayeredPerceptron:

    def __init__(self, inputs, hiddenlayers, outputs):

        self.inputs = inputs
        self.hiddenlayers = hiddenlayers
        self.outputs = outputs

        # Getting the number of layers. Hidden layer/s is an array, each
        # index indicates the amount of neurons on said layer
        # e.g index-0 - number of neurons in input layer
        if hiddenlayers[0] == 0: 
            layers = [inputs] + [outputs]
        else: 
            layers = [inputs] + hiddenlayers + [outputs]

        weights = []
        # Used for momentum
        weights_previous = []

        bias = []
        # Previous Bias
        bias_previous = []
        # Assuming a fully connected NN between layer i and layer i+1  
        for i in range(len(layers) - 1):
            # We create a 2D array of random weights
            # The number of rows is equal to number of neurons in layer i
            # The number of columns is equal to number of neurons in layer i+1
            w = 2 * np.random.rand(layers[i], layers[i+1]) - 1
            # Weights is a list with 2D matrices of weights as items
            # Amount of those 2D matrices will be layers - 1 (edges of NN)

            # weights for bias for the next layer
            bias_w = 2 * np.random.rand(layers[i+1]) - 1

            weights.append(w)
            weights_previous.append(w)
            bias.append(bias_w)
            bias_previous.append(bias_w)

        self.weights = weights
        self.weights_previous = weights_previous
        self.bias = bias
        self.bias_previous = bias

        # List of arrays of the activations of each layer
        activations = []
        for i in range(len(layers)):
            # We initialize with zero just to create the structure
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

         # List of arrays of the deltas of each layer
        deltas = []
        for i in range(len(layers)-1):
            # We initialize with zero just to create the structure
            d = np.zeros(layers[i])
            deltas.append(d)
        self.deltas = deltas

    # 2 steps:
    # Summing Junction - dot product of inputs and weights and so on.
    # Activation Function - sigmoid
    def forward_propagation(self, inputs):
        activations = inputs
        self.activations[0] = activations

        # Go through all the weights going from left to right layer
        for i, w in enumerate(self.weights):
            # Calculate the net inputs - dot product between
            # input with weights at first and then the activations
            # of neurons with the weights going to the next layer
            net_inputs = np.dot(activations, w) + self.bias[i]*1
            activations = self.sigmoid(net_inputs)
            self.activations[i+1] = activations

        # The last activation is also my MLP output
        return activations

    def back_propagation(self, error):

        # Going from right to left
        # Minus one because input layer doesn't have deltas
        for i in reversed(range(len(self.activations) - 1)):
            # i+1 in combination with reversed basically skips the first index (input layer)
            # Get the activation
            activations = self.activations[i+1]

            # Calculate Deltas of output layer
            if ((i + 1) == len(self.activations) - 1):
                delta = error * self.sigmoid_derivative(activations)
                self.deltas[i] = delta
            # Calculate Deltas of hidden layers
            else:
                # It's matrix multiplication of deltas rows (which will be 1 value since deltas is 1-D horizontal vector)
                # with the corresponding weights columns
                delta = np.dot(self.deltas[i+1], self.weights[i+1].T) * self.sigmoid_derivative(activations)
                self.deltas[i] = delta

    # Process to update my weights
    def gradient_descent(self, learning_rate, momentum):
        for i in range(len(self.weights)):
            # Obtain weights, deltas, activations and bias for a layer
            weights = self.weights[i]
            deltas = self.deltas[i]
            activation = self.activations[i]
            bias = self.bias[i]

            # Adaptation of weights

            # Basically we want for each weight[i, j] to use delta[j] and activations[i]
            self.weights[i] = weights + learning_rate * np.outer(activation, deltas) + momentum * (weights - self.weights_previous[i])
            # Updating our bias
            self.bias[i] = bias + learning_rate * deltas + momentum * (bias - self.bias_previous[i])

            self.weights_previous[i] = weights
            self.bias_previous[i] = bias
            
    def train(self, train_inputs, train_targets, learning_rate, momentum):

            sum_error = 0
            success_num = 0

            # Take 1 input data and the corresponding output target every time
            for j, input in enumerate(train_inputs):

                target = train_targets[j]
                
                # Perform forward propagation
                output = self.forward_propagation(input)

                # Calculate error
                error = target - output

                # We obtain the 26bit encoding 
                output_encoding = self.output_to_ohe(output)

                # Check if we had a correct prediction - only need to compare indexes of 1
                if (np.argmax(output_encoding) == np.argmax(target)):
                    success_num += 1

                # Perform back propagation
                self.back_propagation(error)

                # Applying gradient descent
                self.gradient_descent(learning_rate, momentum)

                sum_error += self.mse(target, output)

            return sum_error / len(train_inputs), success_num / len(train_inputs)

    def test(self, test_inputs, test_targets):

        sum_error = 0
        success_num = 0

        # Take 1 input data and the corresponding output target every time
        for j, input in enumerate(test_inputs):

            target = test_targets[j]
            
            # Perform forward propagation
            output = self.forward_propagation(input)

            # We obtain the 26bit encoding 
            output_encoding = self.output_to_ohe(output)

            # Check if we had a correct prediction - only need to compare indexes of 1
            if (np.argmax(output_encoding) == np.argmax(target)):
                success_num += 1

            sum_error += self.mse(target, output)

        return sum_error / len(test_inputs), success_num / len(test_inputs)    

    # Function that calls training and testing
    # Also responsible for writing the results into the error and success rate file
    def train_testing_calls(self, train_inputs, train_targets, epochs, learning_rate, momentum, filename, test_inputs, test_targets):

            error_file_output = ""
            successrate_file_output = ""

            for i in range(epochs):

                print("Epoch: %2d" % (i+1))

                train_error, train_success = NN.train(train_inputs, train_targets, learning_rate, momentum)

                test_error, test_success = NN.test(test_inputs, test_targets)
                
                # Concatenate the results of all our epochs

                error_file_output += str(i) + " " + str(train_error) + " " + str(test_error) + "\n"

                successrate_file_output += str(i) + " " + str(train_success) + " " + str(test_success) + "\n"

            # Write into errors.txt file
            error_file = filename.joinpath("errors.txt")
            with open(error_file, "w") as file:
                file.write(error_file_output)

            # Write into successrate.txt file
            successrate_file = filename.joinpath("successrate.txt")
            with open(successrate_file, "w") as file:
                file.write(successrate_file_output)

    # Mean square error
    def mse(self, target, output):
        return np.average((output - target)**2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    # Output to one hot encoding
    def output_to_ohe(self, output):

        output_df = pd.DataFrame(output)
                
        # Replace the max number with 1 and all others with 0 in dataframe output_df
        output_df = output_df.replace(output_df.max(), 1)

        return output_df

# Class used for reading from files
class Reader:

    # Function for reading from our parameter.txt file
    def readParameters(filename):
        # Open the file
        with open(filename, "r") as file:
            # Get and return the parameters as an array
            return [ line.split(' ').pop() for line in file ]

# Class for plotting our gathered error and success rate data
class Plots:

    def create_plots(filename):

        figure, (ax1, ax2) = plt.subplots(1, 2)

        Plots.error_plot(filename, ax1)
        Plots.successrate_plot(filename, ax2)

        plt.show()

    def error_plot(filename, ax1):

        errors_filename = filename.joinpath("errors.txt")

        # Plot for error file
        epochs = np.loadtxt(errors_filename, usecols = (0))
        training_error = np.loadtxt(errors_filename, usecols = (1))
        test_error = np.loadtxt(errors_filename, usecols = (2))

        ax1.set_title("Error File")
        ax1.set(xlabel = 'Epochs', ylabel='Error')
        ax1.plot(epochs, training_error, color = 'r', label= 'Training Error')
        ax1.plot(epochs, test_error, color = 'b', label= 'Test Error')
        ax1.legend()

    def successrate_plot(filename,ax2):

        sr_filename = filename.joinpath("successrate.txt") 

        # Plot for success rate file
        epochs = np.loadtxt(sr_filename, usecols = (0))
        train_success = np.loadtxt(sr_filename, usecols = (1))
        test_success = np.loadtxt(sr_filename, usecols = (2))

        ax2.set_title("SuccessRate File")
        ax2.set(xlabel='Epochs', ylabel='Success rate percentage')
        ax2.plot(epochs, train_success, color ='r', label = 'Training Success Rate')
        ax2.plot(epochs, test_success, color = 'b', label = 'Testing Success Rate')
        ax2.legend()

# Used for one hot encoding of output
class data_manipulation():
    # Encode our letter outputs to 26bit values
    # e.g A = 1000... , B = 0100... etc
    def encoding_outputs(outputs_target):

        # One hot encoding of letters
        outputs_target = pd.get_dummies(outputs_target)
        outputs_target = outputs_target.to_numpy()
        outputs_target = outputs_target.tolist()

        return outputs_target
       
if __name__ == "__main__":

    # The path used for finding the files. All files need to be in the same directory
    filename = pathlib.Path(__file__).parent

    # Obtaining the parameters from our parameter file
    p_filename = filename.joinpath("parameters.txt")
    parameters = Reader.readParameters(p_filename)

    # Initializing our variables
    numHiddenLayerOneNeurons = int(parameters[0])
    numHiddenLayerTwoNeurons = int(parameters[1])
    numInputNeurons = int(parameters[2])
    numOutputNeurons = int(parameters[3])
    learningRate = float(parameters[4])
    momentum = float(parameters[5])
    maxIterations = int(parameters[6])
    trainFile = parameters[7].strip()
    testFile = parameters[8].strip()

    # Obtaining training inputs and outputs
    training_filename = filename.joinpath(trainFile)
    train_inputs = np.loadtxt(training_filename, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
    train_outputs_target = pd.read_csv(training_filename, usecols = [0], sep=" ", header=None)

    # Obtaining testing inputs and outputs
    test_filename = filename.joinpath(testFile)
    test_inputs = np.loadtxt(test_filename, usecols= (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
    test_outputs_target = pd.read_csv(test_filename, usecols = [0], sep=" ", header=None)

    train_outputs_target = data_manipulation.encoding_outputs(train_outputs_target)
    test_outputs_target = data_manipulation.encoding_outputs(test_outputs_target)

    # Create a NN
    if numHiddenLayerTwoNeurons == 0:
        NN = NeuralNetwork_MultiLayeredPerceptron(numInputNeurons, [numHiddenLayerOneNeurons] , numOutputNeurons)
    else:
        NN = NeuralNetwork_MultiLayeredPerceptron(numInputNeurons, [numHiddenLayerOneNeurons, numHiddenLayerTwoNeurons] , numOutputNeurons)

    # Train and Test NN
    NN.train_testing_calls(train_inputs, train_outputs_target, maxIterations, learningRate, momentum, filename, test_inputs, test_outputs_target)

    # Plots
    Plots.create_plots(filename)