from __future__ import print_function, division
import numpy as np
import math
from mlfromscratch_mine.utils import normalize, polynomial_features

class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1)

    def grad(self, w):
        return self.alpha * np.sign(w)

# alpha = regularization strength
# regularization = weight가 너무 커지는 것을 억제해서 overfitting을 줄이는 기법
class l2_regularization():
    """ Regularization for Ridge Regression """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w) #1/2는 미분할때 2 상쇄

    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w, ord=1)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)

class Regression(object):
    """ 
    Base regression model.
    Models the relationship between a scalar dependent variable y and the independent variable x
    Params: n_iterations, learning_rate
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1) # X라는 배열에 index 0에 value 1값을 column축으로 넣어라
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            # Forward
            y_pred = X.dot(self.w)
            # Loss
            data_loss = np.mean(0.5 * (y - y_pred)**2)
            reg_loss = self.regularization(self.w[1:])
            total_loss = data_loss + reg_loss
            self.training_errors.append(total_loss)
            # Gradient
            grad_w = -X.T.dot(y - y_pred) / len(y) 
            grad_w[1:] += self.regularization.grad(self.w[1:])
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        # If not gradient descent => Least sequares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, Vt = np.linalg.svd(X.T.dot(X)) # Left Singular Vectors, Diagonal Matrix, Right Singular Vectors
            S_inv = np.linalg.pinv(np.diag(S))
            X_sq_reg_inv = Vt.T.dot(S_inv).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y) # w = (X.T * X)^-1 * X.T * y
        else:
            super(LinearRegression, self).fit(X, y)

class LassoRegression(Regression):
    """ 
    Linear regression model with a regularization factor which does both variable selection
    and regularization. Model that tries to balance the fit of the model with respect to the training
    data and the complexity of the model. A large regularization factor with decreases the variance of
    model
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)

class PolynomialRegression(Regression):
    """ 
    Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression
    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)

class RidgeRegression(Regression):
    """ Also referred to as Tikhonov regularization """
    def __init__(self, reg_factor, n_iterations=3000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, learning_rate)

class PolynomialRidgeRegression(Regression):
    """ Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)

class ElasticNet(Regression):
    """ 
    Regression where a combination of l1 and l2 regularization are used.
    The ratio of their contributions are set with the 'l1_ratio' parameter.
    """
    def __init__(self, degree=1, reg_factor=0.5, l1_ratio=0.5, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)