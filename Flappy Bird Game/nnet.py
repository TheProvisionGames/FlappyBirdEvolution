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

    def modify_weight(self):
        """ Function to modify weights of the bird
        """

        # Modify the weights from the input to the hidden layer
        Nnet.modify_array(self.weight_input_hidden)

        # Modify the weights from the hidden to the output layer
        Nnet.modify_array(self.weight_hidden_output)

    def create_mixed_weights(self, net1, net2):
        """ Function to mix weights of two birds

        :param net1: class, containing neural network information
        :param net2: class, containing neural network information
        """

        # Mix the weights from the input to the hidden layer
        self.weight_input_hidden = Nnet.get_mix_from_arrays(net1.weight_input_hidden, net2.weight_input_hidden)

        # Mix the weights from the hidden to the output layer
        self.weight_hidden_output = Nnet.get_mix_from_arrays(net1.weight_hidden_output, net2.weight_hidden_output)

    def modify_array(a):
        """ Function to modify a weight array

        :param a: numpy array, containing floats representing weights
        """

        # Use a multi-dimensional iterator to read all values in the array
        for x in np.nditer(a, op_flags=['readwrite']):

            # Generate a random float number and check if this is lower than the set modification chance
            if random.random() < MUTATION_WEIGHT_MODIFY_CHANCE:
                # Change the weight with a random float numbers from [-5, 0)
                x[...] = np.random.random_sample() - 0.5

    def get_mix_from_arrays(ar1, ar2):
        """

        :param ar1: numpy array, 2-dimensional containing floats numbers representing weights
        :param ar2: numpy array, 2-dimensional containing floats numbers representing weights
        :return:
           res: numpy array, 2-dimensional containing floats numbers representing weights
        """

        # Get the total number of weights
        total_entries = ar1.size
        # Get the number of row weights
        num_rows = ar1.shape[0]
        # Get the number of column weights
        num_cols = ar1.shape[1]

        # Define how many weights should be modified
        num_to_take = total_entries - (int(total_entries * MUTATION_ARRAY_MIX_PERC))
        # Set the index numbers of the weights to modify
        idx = np.random.choice(np.arange(total_entries), num_to_take, replace=False)

        # Fill a new array with random weight values
        res = np.random.rand(num_rows, num_cols)

        # For every weight
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                # Get the index position of the specific weight
                index = row * num_cols + col

                # Check if the weight should be changed by checking the index
                if index in idx:
                    # Get the weight of the first parent when the weight should be changed
                    res[row][col] = ar1[row][col]
                # Else get the weight of the second parent
                else:
                    res[row][col] = ar2[row][col]

        # Return the array having weights of the child bird
        return res

