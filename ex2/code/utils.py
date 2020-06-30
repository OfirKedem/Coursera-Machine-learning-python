import numpy as np
import pandas as pd


# NOTE:
#   m =  # data points
#   n =  # features


def sig(x):
    """
    computes element wise sigmoid function

    :param x: scalar or array of any shape
    :type x: float, np.ndarray
    :return: array of the same shape of x, containing the output of sigmoid function
     for the corresponding entry in x. i.e. out[i,j] = sigmoid(x[i, j])
    :rtype: float, np.ndarray
    """
    return 1 / (1 + np.exp(-x))


def get_data(data):
    """
    get data from DataFrame, last column is the labels.
    adds column of ones to x, as first column

        :param data: each row contain the features, last column is the labels
        :type data: pd.DataFrame
        :return: (x,y) x contain the data, shape (m,n+1).
                y contains the labels, shape (m, 1)
        :rtype: tuple
    """
    x = np.append(np.ones([data.shape[0], 1]), data.iloc[:, :-1], axis=1)
    y = data.iloc[:, -1].values
    return x, y


def cost_function(theta, x, y, lamb=0.0):
    """computes binary logistic regression loss function

    :param theta: model parameters / weights, shape (n+1, )
    :param x: data, shape (m,n+1)
    :param y: labels, shape (m, )
    :param lamb: regularization parameter lambda, scalar. default = 0.0.
    :type theta: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type lamb: float
    :return: the total cost, scalar
    :rtype: float
    """

    h = sig(x.dot(theta))
    cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() / len(y)
    reg = (theta[1:] ** 2).sum() * lamb / (2 * len(y))
    return cost + reg


def compute_grad(theta, x, y, lamb=0.0):
    """computes binary logistic regression gradient

        :param theta: model parameters / weights, shape (n+1, )
        :param x: data, shape (m,n+1)
        :param y: labels, shape (m, )
        :param lamb: regularization parameter lambda, scalar. default = 0.0.
        :type theta: np.ndarray
        :type x: np.ndarray
        :type y: np.ndarray
        :type lamb: float
        :return: gradient of loss function with respect to the parameters, shape (n+1, )
        :rtype: float
        """
    h = sig(x.dot(theta))
    cost_grad = (h - y).dot(x) / len(y)
    reg_grad = (lamb / len(y)) * theta
    reg_grad[0] = 0  # bais weight is not regularized
    return cost_grad + reg_grad


def map_feature(x, poly_deg=6):
    """
    creates all polynomial features up to poly_deg.

    :param x: data with ones in the first column and two more features, shape(m, 3)
    :param poly_deg: the maximum polynomial degree, scalar
    :type x: np.ndarray
    :type poly_deg: int
    :return: a new data matrix with all the possible polynomial features up to poly_deg, shape(m, #possible_combination)
    :rtype: np.ndarray
    """
    # compute the number of possible combinations.
    new_dim = np.arange(1, poly_deg + 2).sum()

    new_x = np.zeros([x.shape[0], new_dim])
    pos = 0

    for i in range(poly_deg + 1):
        for j in range(poly_deg - i + 1):
            new_x[:, pos] = (x[:, 1] ** i) * (x[:, 2] ** j)
            pos += 1
    return new_x
