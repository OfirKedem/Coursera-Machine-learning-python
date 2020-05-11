import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# NOTE:
#   m =  # data points
#   n =  # features


# 1 Regularized Linear Regression
def add_bais(x):
    """
    adds a column of ones, as the first column

    :param x: data, shape (m,n)
    :type x: np.ndarray
    :return: data with a column if ones, shape (m, n+1)
    :rtype: np.ndarray
    """
    return np.append(np.ones([x.shape[0], 1]), x, axis=1)


def linear_reg_cost_function(theta, x, y, lamb=0.0):
    """
    compute MSE for linear regression with regularization

    :param theta: parameters, shape(n+1, )
    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param lamb: regularization parameter lambda, scalar.
    :type theta: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type lamb: float
    :return: MSE with regularization
    :rtype: float
    """
    cost = (x.dot(theta) - y).dot((x.dot(theta) - y))/(2 * len(y))
    reg = (lamb / (2 * len(y))) * theta[1:].dot(theta[1:])
    return cost + reg


def compute_grad(theta, x, y, lamb=0):
    """
    compute MSE loss linear regression gradient with regularization

    :param theta: parameters, shape(n+1, )
    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param lamb: regularization parameter lambda, scalar.
    :type theta: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type lamb: float
    :return: gradient vector, shape (n+1, )
    :rtype: np.ndarray
    """
    cost_grad = (x.dot(theta) - y).dot(x) / len(y)
    reg_grad = (lamb / len(y)) * theta
    reg_grad[0] = 0  # no regularization on bais
    return cost_grad + reg_grad


def fit(init_theta, x, y, lamb=0.0, method='L-BFGS-B'):
    """
    Fit the model to the data with parameter lambda.

    A rapper of the sscipy.optimize.minimize function.

    :param init_theta: parameters, shape(n+1, )
    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param lamb: regularization parameter lambda, scalar.
    :param method: the method of minimization
    :type init_theta: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type lamb: float
    :type method: str or None
    :return: learned parameters, shape (n+1, )
    :rtype: np.ndarray
    """
    op_result = minimize(linear_reg_cost_function, init_theta, args=(x, y, lamb), jac=compute_grad, method=method)
    if op_result.success:
        print('Optimization terminated successfully.')
    else:
        print('Optimization failed!!!')
        raise RuntimeError(op_result.message)
    return op_result.x


# 2 Bias-variance
def learning_curve(x, y, xval, yval, lamb=0.0):
    """
    computes train and validation error on different train set sizes.

    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param xval: validation set data, shape(m_val, n+1)
    :param yval: validation set labels, shape (m_val, )
    :param lamb: regularization parameter lambda, scalar.
    :type x: np.ndarray
    :type y: np.ndarray
    :type xval: np.ndarray
    :type yval: np.ndarray
    :type lamb: float
    :return: arrays of training error and validation error
    :rtype: (np.ndarray, np.ndarray)
    """
    m = len(y)
    train_err = np.zeros(m)
    val_err = np.zeros(m)
    init_theta = np.ones(x.shape[1])
    for i in range(m):
        xtrain = x[:(i+1)]
        ytrain = y[:(i+1)]
        theta = fit(init_theta, xtrain, ytrain, lamb=lamb)
        train_err[i] = linear_reg_cost_function(theta, xtrain, ytrain)
        val_err[i] = linear_reg_cost_function(theta, xval, yval)
    return train_err, val_err


def plot_learning_curve(x, y, xval, yval, lamb=0.0):
    """
    computes and plots the learning curves

    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param xval: validation set data, shape(m_val, n+1)
    :param yval: validation set labels, shape (m_val, )
    :param lamb: regularization parameter lambda, scalar.
    :type x: np.ndarray
    :type y: np.ndarray
    :type xval: np.ndarray
    :type yval: np.ndarray
    :type lamb: float
    :return: None
    """
    train_err, val_err = learning_curve(x, y, xval, yval, lamb)
    plt.plot(range(1, len(y)+1), train_err, label='Train')
    plt.plot(range(1, len(y)+1), val_err, label='Cross Validation')
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()


def poly_features(x, p):
    """
    adds polynomial features up to p degree. x already has a column of ones

    :param x: data, shape(m, n+1)
    :param p: max polynomial degree, scalar
    :type x: np.ndarray
    :type p: int
    :return: new data matrix with polynomial features, shape(m, p+1)
    :rtype: np.ndarray
    """

    if p < 2:
        return x

    m = x.shape[0]
    x_poly = np.zeros([m, p + 1])
    x_poly[:, :2] = x
    for i in range(2, p + 1):
        x_poly[:, i] = x_poly[:, 1] ** i
    return x_poly


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
    x[:, 1:] = (x[:, 1:]-mu)/sigma
    return x


def validation_curve(x, y, xval, yval, l_vec):
    """
    compute validation and train error for different lambda's

    :param x: data, shape (m, n+1)
    :param y: labels, shape (m, )
    :param xval: validation set data, shape(m_val, n+1)
    :param yval: validation set labels, shape (m_val, )
    :param l_vec: vector of different regularization parameter lambda
    :type x: np.ndarray
    :type y: np.ndarray
    :type xval: np.ndarray
    :type yval: np.ndarray
    :type l_vec: np.ndarray
    :return: arrays of training error and validation error
    :rtype: (np.ndarray, np.ndarray)
    """
    m = len(l_vec)
    train_err = np.zeros(m)
    val_err = np.zeros(m)
    init_theta = np.ones(x.shape[1])
    for i, l in enumerate(l_vec):
        theta = fit(init_theta, x, y, l)
        train_err[i] = linear_reg_cost_function(theta, x, y)
        val_err[i] = linear_reg_cost_function(theta, xval, yval)
    return train_err, val_err


