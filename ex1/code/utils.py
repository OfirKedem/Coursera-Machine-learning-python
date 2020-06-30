import numpy as np
from pandas import DataFrame


# NOTE:
#   m =  # data points
#   n =  # features


def get_data(data):
    """
    get data from DataFrame, last column is the labels.
    adds column of ones to x, as first column

        :param data: each row contain the features, last column is the labels
        :type data: DataFrame
        :return: (x,y) x contain the data, shape (m,n+1).
                y contains the labels, shape (m, 1)
        :rtype: tuple
    """
    x = np.append(np.ones([data.shape[0], 1]), data.iloc[:, :-1], axis=1)
    y = np.expand_dims(data.iloc[:, -1].values, axis=1)
    return x, y


def compute_cost(x, y, theta):
    """
    computes MSE of linear model

    :param x: data. data points in the rows, features in columns, shape (m,n+1)
    :param y: labels, shape (m,1)
    :param theta: weights, shape (n+1,1)
    :type x: np.ndarray
    :type y: np.ndarray
    :type theta: np.ndarray
    :return: MSE:  j = (((x.dot(theta) - y)**2).sum()) / (2 * len(y))
    :rtype: float
    """
    j = (((x.dot(theta) - y) ** 2).sum()) / (2 * len(y))
    return j


def gradient_descent(x, y, init_theta, alpha, iterations=50):
    """
    a batch GD implementation

    :param x: data. data points in the rows, features in columns, shape (m,n+1)
    :param y: labels, shape (m,1)
    :param init_theta: weights, shape (n+1,1)
    :param alpha: learning rate
    :param iterations: # of iterations

    :type x: np.ndarray
    :type y: np.ndarray
    :type init_theta: np.ndarray
    :type alpha: float
    :type iterations: int

    :returns: (theta, j_vec)
            theta: learned weights, shape (n+1,1)
            j_vec:  j values across the training, shape (iterations, )
    :rtype: (np.ndarray, np.ndarray)
    """
    theta = init_theta
    m = len(y)
    j_vec = np.zeros(iterations)
    for itr in range(iterations):
        j_vec[itr] = compute_cost(x, y, theta)
        theta -= (alpha / m) * (x.dot(theta) - y).T.dot(x).T
    return theta, j_vec


def normalize(x, mu, sigma):
    """
    normalize features by a specific mean and std, except first column of ones

    :param x: data to normalize, shape(_,n+1)
    :param mu: mean, shape(n, )
    :param sigma: std, shape(n, )
    :type x: np.ndarray
    :type mu: np.ndarray
    :type sigma: np.ndarray
    :return: normalize data
    :rtype: np.ndarray
    """
    if len(x.shape) == 1:  # in case of 1D array
        x[1:] = (x[1:] - mu) / sigma
    else:
        x[:, 1:] = (x[:, 1:] - mu) / sigma
    return x
