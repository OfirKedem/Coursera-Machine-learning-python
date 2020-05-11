import numpy as np
from scipy.optimize import minimize

# NOTE:
#   m =  # data points
#   n =  # features


def sig(x):
    """
    computes element wise sigmoid function

    :param x: scalar or array of any shape
    :type x: float, np.ndarray
    :return: 1/(1+np.exp(-x))
    :rtype: float, np.ndarray
    """
    return 1/(1+np.exp(-x))


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
    cost = (-y * np.log(h) - (1-y) * np.log(1 - h)).sum() / len(y)
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
    cost_grad = (h-y).dot(x)/len(y)
    reg_grad = (lamb/len(y))*theta
    reg_grad[0] = 0
    return cost_grad + reg_grad


def one_vs_all(init_theta_mat, x, y, k, lamb=0.0):
    """
    trains k linear logistic regression classifiers in the one vs all method

    :param init_theta_mat: matrix containing the initial parameters for each classifier, shape (k, n+1)
    :param x: data, shape (m,n+1)
    :param y: labels containing numbers in range(k), shape (m, )
    :param k: number of classes, scalar
    :param lamb: regularization parameter lambda, scalar. default = 0.0.
    :type init_theta_mat: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type k: int
    :type lamb: float
    :return: the learned parameters for each classifier, shape (k, n+1)
    :rtype: np.ndarray
    """
    theta_mat = np.zeros_like(init_theta_mat)
    for i in range(k):
        tmp_y = (y == i) * 1
        op_result = minimize(cost_function, init_theta_mat[i, :],
                             args=(x, tmp_y, lamb), jac=compute_grad, method='L-BFGS-B')

        if not op_result.success:
            print("something went wrong!!!")
            print(op_result.message)

        theta_mat[i, :] = op_result.x
    return theta_mat

