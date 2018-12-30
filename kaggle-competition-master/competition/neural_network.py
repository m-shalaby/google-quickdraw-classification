import numpy as np
from numpy.random import RandomState
import numpy.linalg as linalg

class NeuralNetwork():
    """
    This class contains our own implementation of the neural network
    """

    def __init__(self, input_size, output_size, rate, number_node_per_layer, number_layer, seed=None):
        """
        Build a feed forward neural network using the given hyperparameters.
        Note: We should find a way to add more activation functions
        Additional information:
        each weight matrix is as follows : i input nodes, j output nodes, w_ij is the weight from node i to
        node j. The resulting matrix is i x j in size.
        The stored values are in matrices (1, n)
        """
        self.learning_rate = rate
        self.node_per_layer = number_node_per_layer
        self.layer_count = number_layer
        self.input_count = input_size + 1       # Take into account the bias term
        self.output_count = output_size
        self.random_state = RandomState(seed)

        # Let W be the weight matrix where w_ij is the weight connecting node i to node j. Each layer has a weight matrix.
        # For each layer (except output), create a weight matrix

        self.weights = []
        self.__ff_storage = []
        self.__stored_weights = []
        if(self.layer_count == 0):
            self.weights = [self.random_state.rand(self.input_count, self.output_count)]

        else:
            for i in range(0, self.layer_count + 1):
                #Initialize weights : could be also a hyperparameter

                if( i == 0):
                    W = self.random_state.rand(self.input_count, self.node_per_layer)

                elif( i == self.layer_count):
                    W = self.random_state.rand(self.node_per_layer, self.output_count)

                else:
                    W = self.random_state.rand(self.node_per_layer, self.node_per_layer)
                
                self.weights.append(W)

        self.__init_stored_weights()
            
    def __init_stored_weights(self):
        self.__stored_weights = []

        for i in range(0, len(self.weights)):
            new_weights = np.zeros(self.weights[i].shape)
            self.__stored_weights.append(new_weights)

    def __feed_forward(self, input:np.ndarray):
        """
        Assume input has the same size has self.input_size - 1
        Each hidden layer stores the evaluated value in a list
        """

        input = np.append(input, 1)
        layer_input = input

        self.__ff_storage = []

        self.__ff_storage.append(layer_input)

        for i in range(0, len(self.weights)):
            layer_weights = self.weights[i]                                         # layer_weights is a matrix
            preactivation = np.dot(layer_input, layer_weights)   
            layer_output = np.apply_along_axis(self.sigmoid, 0, preactivation)       # Apply sigmoid function over all the nodes
            layer_input = layer_output

            self.__ff_storage.append(layer_output)                              

        return layer_input

    def __back_propagation(self, errors, error_derivatives, batch_update=True):
        """
        Update the weights
        """
        layer_input = np.transpose(np.atleast_2d(error_derivatives))

        for i in range(len(self.weights)-1, -1, -1):
            stored_values = self.__ff_storage[i + 1]

            layer_weights = self.weights[i]

            prev_layer_output = self.__ff_storage[i]

            D = np.diag(np.apply_along_axis( self.sigmoid_derivative, 0, stored_values))
            o = np.atleast_2d(prev_layer_output)
            delta = np.dot(D, layer_input)
            # Update the weights using the stored weights + current update
            if batch_update == True:
                self.weights[i] = self.weights[i] - self.learning_rate * np.transpose(np.dot(delta, o)) - self.__stored_weights[i]
            else:
                self.__stored_weights[i] = self.__stored_weights[i] + self.learning_rate * np.transpose(np.dot(delta, o))
            layer_input = np.dot(layer_weights, delta)

    def sigmoid(self, x:float) -> float:
        return  ( 1.0 / (1.0 + np.exp(-1.0 * x)))

    def sigmoid_derivative(self, sig:float) -> float:
        """
        Compute the derivate of the sigmoid from the sigmoid itself
        """
        return sig * (1 - sig)

    def quadratic_error(self, output, expected_output):
        """
        Returns the error and its derivative for the given output and expected output
        """
        errors = []
        derivatives = []
        for i in range(0, len(output)):
            diff = (output[i] - expected_output[i])
            errors.append(0.5 *  np.power(diff ,2))
            derivatives.append(diff)

        return np.array(errors), np.array(derivatives)

    def train(self, input, expected_output):
        for i in range(0, len(input)):
            output = self.__feed_forward(input[i])
            error, derivative = self.quadratic_error(output, expected_output[i])
            self.__back_propagation(error, derivative) 
        
    def train_batch(self, input, expected_output, batch_size):
        """
        Assumes batch size is a multiple of the total size
        """
        number_batch = int(len(input) / batch_size)

        for b in range(0, number_batch):

            #reset memory of updates
            self.__init_stored_weights()

            for i in range(0, batch_size):
                index = i + batch_size * b
                output = self.__feed_forward(input[index])
                error, error_derivative = self.quadratic_error(output, expected_output[index])

                # perform partial backpropagation for all indices except the last one. Partial meaning computing weights,
                # storing them for later but no update to actual weights

                if( i == batch_size - 1):
                    self.__back_propagation(error, error_derivative, True)
                else:
                    self.__back_propagation(error, error_derivative, False)
            
    def predict(self, input):
        """
        Predict the result of an array of input
        """
        pred = []
        for i in range(0, len(input)):
            pred.append(self.__feed_forward(input[i]))
        return np.array(pred)

def XOR_test():
    test = NeuralNetwork(2, 1, 0.2, 4, 1, seed=0)
    inputs, outputs = build_XOR_dataset()
    
    test.train(inputs, outputs)

    test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1] ])
    print(test.predict(test_input))
    #print(str(test.predict(np.array([0, 0]))))      #[0.02898488]
    #print(str(test.predict(np.array([1, 1]))))      #[0.06952503]
    #print(str(test.predict(np.array([1, 0]))))      #[0.94180268]
    #print(str(test.predict(np.array([0, 1]))))      #[0.93965125]

def build_XOR_dataset():
    inputs = []
    outputs = []

    for i in range(0, 5000):
        inputs.append(np.array([0, 0]))        
        inputs.append(np.array([1, 1]))
        inputs.append(np.array([1, 0]))
        inputs.append(np.array([0, 1]))

        outputs.append(np.array([0]))
        outputs.append(np.array([0]))
        outputs.append(np.array([1]))
        outputs.append(np.array([1]))


    rng = RandomState(0)

    rng.shuffle(inputs)
    rng = RandomState(0)
    rng.shuffle(outputs)

    return inputs, outputs

def XOR_batch_test(batch_size):
    inputs, outputs = build_XOR_dataset()
    test = NeuralNetwork(2, 1, 0.2, 4, 1, seed=0)

    test.train_batch(inputs, outputs, batch_size)
    
    test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1] ])
    print(test.predict(test_input))

if __name__ == '__main__':
    #XOR_test()
    XOR_batch_test(32)
