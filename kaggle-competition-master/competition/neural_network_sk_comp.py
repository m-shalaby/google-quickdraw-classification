"""
Use the sklearn framework to use GridSearch with a custom estimator.
Warning : this version of the neural network is only good for classfication on output by taking the argmax of the prediction
"""
import neural_network
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend

from numpy.random import RandomState

class CustomNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, batch_size=10, node_per_layer=4, layer_count=1, learning_rate=0.01,epoch=1, seed=123):
        self.batch_size = batch_size
        self.node_per_layer = node_per_layer
        self.layer_count = layer_count
        self.learning_rate = learning_rate
        self.seed=seed
        self.epoch = epoch
        #harcoded sizes for 28x82 images on 31 classes
        self.input_size = 784
        self.output_size = 31

        self.NN = neural_network.NeuralNetwork(self.input_size, self.output_size, self.learning_rate, self.node_per_layer, self.layer_count, self.seed)
        
    def fit(self, X, y):

        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        #self.classes_ = unique_labels(y)
        #convert class index to one-hot encoding:

        y_ohe = []

        for i in range(0, len(y)):
            y_ohe.append(np.array([0]*self.output_size))
            y_ohe[i][y[i]] = 1

        self.X_ = X
        self.y_ = y

        for i in range(0, self.epoch):
            self.NN.train_batch(self.X_, y_ohe, self.batch_size)
            # compute new learning rate
            self.learning_rate = self.learning_rate / 2.0

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        pred = self.NN.predict(X)

        #convert one hot encoding to class index:

        final_pred = []
        for i in range(0, len(X)):
            final_pred.append([np.argmax(pred[i])])
        final_pred = np.array(final_pred)
        
        return final_pred

    def get_params(self, deep=True):
        return {"batch_size": self.batch_size, "node_per_layer": self.node_per_layer,
         "layer_count": self.layer_count, "learning_rate": self.learning_rate, "epoch": self.epoch}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.NN = neural_network.NeuralNetwork(self.input_size, self.output_size, self.learning_rate, self.node_per_layer, self.layer_count, self.seed)
        return self

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

    return inputs, outputs #np.array(outputs) #.ravel()

def baseline_custom_neural_network():

    parameters = {"batch_size": [10],
                "node_per_layer": [4],
                "layer_count": [2, 1], 
                "learning_rate": [0.4, 0.2]}

    inputs, outputs = build_XOR_dataset()
    test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1] ])

    grid = GridSearchCV(CustomNNClassifier(), parameters, refit=True, cv=3, verbose=5, return_train_score=True)

    with parallel_backend('threading'):
        grid.fit(inputs, outputs)

    best_parameters = grid.best_params_

    pred = grid.predict(test_input)
    print("Best score for best parameters:")
    print(grid.best_score_)
    print(grid.best_params_)

    test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1] ])
    print(grid.predict(test_input))

def XOR_test():
    test = CustomNNClassifier(2, 2, seed=123)
    inputs, outputs = build_XOR_dataset()

    test.set_params(batch_size=10, node_per_layer=4, layer_count=1, learning_rate=0.2)

    test.fit(inputs, outputs)

    test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1] ])

    print(test.predict(test_input))

if __name__ == '__main__':
    #XOR_test()
    baseline_custom_neural_network()

