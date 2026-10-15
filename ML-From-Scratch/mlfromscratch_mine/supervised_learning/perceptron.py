from __future__ import print_function, division
import math
import numpy as np

from mlfromscratch_mine.utils import train_test_split, to_categorical, normalize, accuracy_score
from mlfromscratch_mine.deep_learning.activation_functions import Sigmoid
from mlfromscratch_mine.deep_learning.loss_functions import SquareLoss
from mlfromscratch_mine.utils.misc import bar_widgets
import progressbar

class Perceptron():
    """ The Percentron. One layer neural network classifier """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        _, n_features = np.shape(X)
        _, n_outputs = np.shape(y)

        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, ))
        self.w0 = np.zeros((1, n_outputs))

        for i in self.progressbar(range(self.n_iterations)):
            # Calculate outputs
            linear_output = X.dot(self.W) + self.w0
            y_pred = self.activation_func(linear_output)
            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss w.r.t each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)
            # Update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0

    def predict(self, X):
        y_pred = self.activation_func(X.dot(self.W) + self.w0)
        return y_pred