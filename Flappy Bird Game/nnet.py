# Import packages
import numpy as np
import scipy.special
import random

# Import all defined values of Flappy Bird variables.
from defs import *

# Create a neural network class
class Nnet:
    def __init__(self, num_input, num_hidden, num_output):
        """ Initialization method for the neural network

        :param num_input: integer, number of input nodes
        :param num_hidden: integer, number of hidden nodes
        :param num_output: integer, number of output nodes
        """

        # Save the number of input, hidden, and output nodes as a property of the class
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # Generate random weights between -0.5 and 0.5 for all connections
        self.weight_input_hidden = np.random.uniform(-0.5, 0.5, size=(self.num_hidden, self.num_input))
        self.weight_hidden_output = np.random.uniform(-0.5, 0.5, size=(self.num_output, self.num_hidden))

        # Define the sigmoid (expit) activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def get_outputs(self, inputs_list):
        """ Function to transform input values to final output values of the neural network

        :param inputs_list: list, containing floats as input values for the neural network
        :return:
            final_outputs: array, two-dimensional containing float numbers representing neural network output values
        """

        # Turn the one dimensional input list in a two dimensional array to use it
        inputs = np.array(inputs_list, ndmin=2).T

        # Multiply the input values by the weights between the input and hidden layer to get the hidden input values
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)

        # Use the activation function to transform the hidden input values
        hidden_outputs = self.activation_function(hidden_inputs)

        # Multiply the hidden output values by the weights between the hidden and output layer to get the final input values of the output nodes
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)

        # use the activation function to transform the input values of the output nodes to final output values
        final_outputs = self.activation_function(final_inputs)

        # Return the two dimensional array of final output values
        return final_outputs

    def get_max_value(self, inputs_list):
        """ Function to call the highest output value of the neural network output array

        :param inputs_list: list, containing floats as input values for the neural network
        :return:
            np.max(outputs): float, highest output value of the neural network
        """

        # Call the function to get a final output array of the neural network
        outputs = self.get_outputs(inputs_list)

        # Get the highest value from the array and return this
        return np.max(outputs)

