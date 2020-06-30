import numpy as np


def sig(x):
    """
    computes element wise sigmoid function

    :param x: scalar or array of any shape
    :type x: float, np.ndarray
    :return: 1/(1+np.exp(-x))
    :rtype: float, np.ndarray
    """
    return 1/(1+np.exp(-x))


# 1.3 Feedforward and cost function
def ff(theta1, theta2, x):
    """
    Feedforward in the neural net.

    :param theta1: first layer weights, shape (hiddenDim, inputDim)
    :param theta2: second layer weights, shape(outputDim, hiddenDim+1)
    :param x: data with first column of ones, shape (m, inputDim)
    :type theta1: np.ndarray
    :type theta2: np.ndarray
    :type x: np.ndarray
    :return: (a1, z2, a2, z3, a3) activation and outputs of each layer
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
#     input layer (x already has a column of ones)
    a1 = x
#     Hidden layer
    z2 = x.dot(theta1.T)
#     activation
    a2 = sig(z2)
#     add bais
    a2 = np.append(np.ones([a2.shape[0], 1]), a2, axis=1)
#     output layer
    z3 = a2.dot(theta2.T)
#     activation
    a3 = sig(z3)
    return a1, z2, a2, z3, a3


def one_hot(y, k):
    """
    turns the label vector to a one-hot matrix.

    :param y: label vector contains values in range(k), shape (m, )
    :param k: number of classes, scalar
    :type y: np.ndarray
    :type k: int
    :return: a one-hot matrix with ones and zeros, shape (m, k)
    :rtype: np.ndarray
    """

#     change label to fit matlab
#     y = (y.astype('int8') - 1) % 10
    y_m = np.zeros([len(y), k])
    y_m[range(len(y)), y] = 1
    return y_m


def roll(x, y):
    """
    puts two matrices of any shapes to a single vector.

    :param x: any shaped array
    :param y: any shaped array
    :type x: np.ndarray
    :type y: np.ndarray
    :return: a vector where its first x.shape[0]*x.shape[1] elements correspond to x elements. the rest are y's.
    :rtype: np.ndarray
    """
    return np.append(x, y)


def unroll(vec, shape1, shape2):
    """
    turns a single vector to two matrices of shapes: shape1 and shape2.

    :param vec: some vector
    :param shape1: what shape to reshape the fist part of vec
    :param shape2: what shape to reshape the last part of vec
    :type vec: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :return: two matrices of shapes: shape1 and shape2
    :rtype: (np.ndarray, np.ndarray)

    .. warning:: shape1[0] * shape1[1] + shape2[0] * shape2[1] must equal to len(vec)
    """
    rows, cols = shape1
    x = np.reshape(vec[:(rows*cols)], shape1)
    y = np.reshape(vec[(rows*cols):], shape2)
    return x, y


def nn_cost_function(theta_vec, x, y, shape1, shape2, lamb=0.0):
    """
    compute the neural net cost function

    :param theta_vec: the flatten parameters of the network.
    :param x: data with first column of ones, shape (m, inputDim)
    :param y: label vector contains values in range(k), shape (m, )
    :param shape1: shape of the first layer parameters (hiddenDim, inputDim)
    :param shape2: shape of the second layer parameters (outputDim, hiddenDim+1)
    :param lamb: regularization parameter lambda, scalar. default = 0.0
    :type theta_vec: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :return: total cost of neural net
    :rtype: float
    """
    theta1, theta2 = unroll(theta_vec, shape1, shape2)

    prob = ff(theta1, theta2, x)  # (m,K)
    prob = prob[-1]
    y_m = one_hot(y, prob.shape[1])  # (m,K)

#     NOTE: The following computation is the trace of a dot product, without explicitly computing the dot product.
#           hence it is much faster. it is equivalent to the following line:
#     cost = np.trace(-yM.dot(np.log(prob).T)-(1-yM).dot(np.log(1-prob).T))
    cost = (-y_m * np.log(prob) - (1 - y_m) * np.log(1 - prob)).sum()
    cost /= len(y)

    reg = ((theta1[:, 1:] ** 2).sum() + (theta2[:, 1:] ** 2).sum()) * (lamb / (2 * len(y)))

    return cost + reg


# 2 Backpropagation
def sig_grad(x):
    """
    computes a element-wise gradient of the sigmoid function at x.

    :param x: scalar or array of any shape
    :type x: float, np.ndarray
    :return: gradient of the same shape as x
    :rtype: float, np.ndarray
    """
    g = sig(x)
    return g * (1 - g)


def rand_initialize_weights(l_in, l_out):
    """
    initialize the weights of a single neural net layer at random

    :param l_in: layer input dim, scalar
    :param l_out: layer output dim, scalar
    :type l_in: int
    :type l_out: int
    :return: random weights, shape (l_out, l_in)
    :rtype: np.ndarray
    """
    eps = np.sqrt(6) / np.sqrt(l_in + l_out)
    return np.random.rand(l_out, l_in) * 2 * eps - eps


def compute_grad(theta_vec, x, y, shape1, shape2, lamb=0, print_loss=False):
    """
    computes gradient of neural net using Backpropagation.

    :param theta_vec: the flatten parameters of the network.
    :param x: data with first column of ones, shape (m, inputDim)
    :param y: label vector contains values in range(k), shape (m, )
    :param shape1: shape of the first layer parameters (hiddenDim, inputDim)
    :param shape2: shape of the second layer parameters (outputDim, hiddenDim+1)
    :param lamb: regularization parameter lambda, scalar. default = 0.0
    :param print_loss: whether or not to print the loss
    :type theta_vec: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :type print_loss: bool
    :return: gradient of all parameters, shape like theta_vec. and loss.
    :rtype: (np.ndarray, float)
    """
    m = len(y)
    k = shape2[0]

    theta1, theta2 = unroll(theta_vec, shape1, shape2)
    y_m = one_hot(y, k)  # (m,outputDim)

    # Feedforward
    # shapes: a1=(m, inputDim), z2=(m,hiddenDim), a2=(m,hiddenDim+1), z3=a3=(m,outputDim)
    a1, z2, a2, z3, a3 = ff(theta1, theta2, x)

    # compute loss, without the cost function because it will call ff again
    prob = a3
    cost = (-y_m * np.log(prob) - (1 - y_m) * np.log(1 - prob)).sum()
    cost /= len(y)
    reg = ((theta1[:, 1:] ** 2).sum() + (theta2[:, 1:] ** 2).sum()) * (lamb / (2 * len(y)))
    total_cost = cost + reg
    if print_loss:
        print(total_cost)

    # compute gradient
    d3 = a3 - y_m  # (m, outputDim)
    theta2_grad = d3.T.dot(a2) / m  # (outputDim, hiddenDim+1)
    d2 = theta2.T.dot(d3.T)[1:, :] * sig_grad(z2.T)  # (hiddenDim, m)
    theta1_grad = d2.dot(a1) / m  # (hiddenDim, inputDim)

    # add regularization term
    theta1_grad[:, 1:] += (lamb / m) * theta1[:, 1:]
    theta2_grad[:, 1:] += (lamb / m) * theta2[:, 1:]

    return roll(theta1_grad, theta2_grad), total_cost


def gd(init_theta_vec, x, y, shape1, shape2, lamb=0, iterations=10000, alpha=0.1, print_loss_every=100):
    """
    a gradient decent algorithm.

    :param init_theta_vec: the flatten initial parameters of the network.
    :param x: data with first column of ones, shape (m, inputDim)
    :param y: label vector contains values in range(k), shape (m, )
    :param shape1: shape of the first layer parameters (hiddenDim, inputDim)
    :param shape2: shape of the second layer parameters (outputDim, hiddenDim+1)
    :param lamb: regularization parameter lambda, scalar.
    :param iterations: number of iterations, scalar
    :param alpha: learning rate, scalar
    :param print_loss_every: prints the loss every some iterations
    :type init_theta_vec: np.ndarray
    :type x: np.ndarray
    :type y: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :type iterations: int
    :type alpha: float
    :type print_loss_every: int
    :return: learned parameters shape like init_theta_vec, and loss vector shape (iterations, )
    :rtype: (np.ndarray, np.ndarray)
    """
    theta_vec = init_theta_vec
    loss = np.zeros(iterations)
    for i in range(iterations):
        grad, loss[i] = compute_grad(theta_vec, x, y, shape1, shape2, lamb=lamb, print_loss=(i % print_loss_every) == 0)
        theta_vec -= alpha*grad
    return theta_vec, loss

