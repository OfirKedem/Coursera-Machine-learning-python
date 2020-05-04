import numpy as np
from pandas import DataFrame


def getData(data: DataFrame):
    """
    get data from datafram, last column is the labels.
    adds column of ones to X, as first column
        :param data: dataframe
        :return: X.shape (data_points, features), y.shape (data_points, 1)
    """
    X = np.append(np.ones([data.shape[0],1]),data.iloc[:, :-1], axis=1)
    y = np.expand_dims(data.iloc[:, -1].values, axis=1)
    return X, y


def computeCost(X, y, theta) -> float:
    """
    computes MSE of linear model

    m - #datapoint

    n - #features

    :param X: data, ndarray of fetures in the columns. shape==(m,n)
    :param y: labels. shape==(m,1)
    :param theta: weights. shape==(n,1)
    :return: MSE, shape==scalar
    """
    J = (((X.dot(theta)-y)**2).sum())/(2*len(y))
    return J


def gradientDescent(X: np.ndarray, y: np.ndarray, init_theta: np.ndarray, alpha: float, iterations: int = 50):
    """
    a batch GD implementation

    m - #datapoint

    n - #features

    :param X: data, ndarray of fetures in the columns. shape==(m,n)
    :param y: labels. shape==(m,1)
    :param init_theta:
    :param alpha: learning rate
    :param iterations: # of iterations
    :return: tuple (theta, lossVec)
    """
    theta = init_theta
    new_theta = init_theta  # value not used
    J = np.zeros(iterations)
    for itr in range(iterations):
        J[itr] = computeCost(X, y, theta)
#         print("%.2f" % J[itr])
        for j in range(len(theta)):
            new_theta[j] = theta[j] - (alpha/len(y))*(X.dot(theta)-y).T.dot(X[:,j])
        theta = new_theta
    return theta, J