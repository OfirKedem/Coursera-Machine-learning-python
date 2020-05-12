import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 1 Anomaly detection

# 1.2 Estimating parameters for a Gaussian
def estimate_gaussian(x):
    """
    estimate gaussian distribution parameters for each column of x

    :param x: data to estimate its column, shape(m,n)
    :type x: np.ndarray
    :return: mean and var for each column of x
    :rtype: (np.ndarray, np.ndarray)
    """
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    return mu, var


def compute_prob(x, mu, var):
    """
    computes the probability of x.

    :param x: the values to compute their probability
    :param mu: the estimated expectation of each column
    :param var: the estimated variance of each column
    :type x: np.ndarray
    :type mu: np.ndarray
    :type var: np.ndarray
    :return: the probability of each entry in x
    :rtype: np.ndarray

    .. note:: assuming that all of x columns are gaussian distributed and uncorrelated.
    """
    norm_factor = 1 / np.sqrt(2 * np.pi * var)
    prob = norm_factor * np.exp(-((x - mu) ** 2) / (2 * var))
    return prob.prod(axis=1)


def visualize_fit(x,  mu, var, res=100):
    """
    plots a contour plot of a uncorrelated gaussian pdf with parameters mu and var.
    plots the data points of x as well.

    :param x: data, shape (m, 2)
    :param mu: the estimated expectation of each column
    :param var: the estimated variance of each column
    :param res: resolution of counter plot
    :type x: np.ndarray
    :type mu: np.ndarray
    :type var: np.ndarray
    :type res: int
    :return: None
    """

    x0_range = np.linspace(x[:, 0].min(), x[:, 0].max(),  res)
    x1_range = np.linspace(x[:, 1].min(), x[:, 1].max(),  res)

    xx, yy = np.meshgrid(x0_range, x1_range)

    x_plot = np.stack([xx.ravel(), yy.ravel()], axis=1)

    prob = compute_prob(x_plot, mu, var)

    z = prob.reshape(xx.shape)

    plt.contour(xx, yy, z, np.logspace(-20, -3, num=10, base=np.e))
    plt.scatter(x[:, 0], x[:, 1], s=2)
    plt.show()


# 1.3 Selecting the threshold
def select_threshold(pval, yval):
    """
    picks the threshold that maximize the f1 score.

    :param pval: probabilities of the data-points in the validation set, shape(m_val, )
    :param yval: labels of validation set (1 if anomaly 0 else), shape (m_val, )
    :type pval: np.ndarray
    :type yval: np.ndarray
    :return: the best threshold and its f1 score
    :rtype: (float, float)
    """
    # all possible values of epsilon to check
    eps_vec = np.linspace(pval.min(), pval.max(), num=1000)

    best_eps = eps_vec[0]
    max_f1 = -1.0
    f1_vec = np.zeros(len(eps_vec))

    for i, eps in enumerate(eps_vec):
        # predict with given epsilon
        pred = (pval < eps) * 1  # times 1 to convert bool to uint

        # compute F1
        tp = ((pred == 1) * (yval == 1)).sum()
        fp = ((yval-pred) == (-1)).sum()
        fn = ((yval-pred) == 1).sum()

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        f1 = 2 * prec * rec / (prec + rec)
        f1_vec[i] = f1

        # keep best epsilon
        if f1 > max_f1:
            best_eps = eps
            max_f1 = f1

    return best_eps, max_f1


# 2 Recommender Systems

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


def compute_cost(parameters, y, r, shape1, shape2, lamb=0.0):
    """
    compute the Collaborative filtering cost function

    :param parameters: the flatten parameters.
    :param y: the ratings of each user to each movie, shape (n_m, n_u)
    :param r: whether or not a specific user rated a specific movie, shape (n_m, n_u)
    :param shape1: shape of the movies parameters matrix
    :param shape2: shape of the users parameters matrix
    :param lamb: regularization parameter lambda, scalar. default = 0.0
    :type parameters: np.ndarray
    :type y: np.ndarray
    :type r: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :return: total cost
    :rtype: float
    """
    x, theta = unroll(parameters, shape1, shape2)

    # compute prediction cost
    pred_cost = (1 / 2) * (((x.dot(theta.T) - y) * r) ** 2).sum()

    # compute regulariztion cost
    reg_cost = (lamb / 2) * (parameters ** 2).sum()

    return pred_cost + reg_cost


def compute_grad(parameters, y, r, shape1, shape2, lamb=0):
    """
    compute the Collaborative filtering gradient of parameters

    :param parameters: the flatten parameters.
    :param y: the ratings of each user to each movie, shape (n_m, n_u)
    :param r: whether or not a specific user rated a specific movie, shape (n_m, n_u)
    :param shape1: shape of the movies parameters matrix
    :param shape2: shape of the users parameters matrix
    :param lamb: optional,  regularization parameter lambda, scalar. default = 0.0
    :type parameters: np.ndarray
    :type y: np.ndarray
    :type r: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :return: gradient of all parameters, shape like parameters.
    :rtype: np.ndarray
    """
    x, theta = unroll(parameters, shape1, shape2)

    # compute prediction gradient
    x_grad = ((x.dot(theta.T) - y) * r).dot(theta)
    theta_grad = ((x.dot(theta.T) - y) * r).T.dot(x)
    pred_grad = roll(x_grad, theta_grad)

    # compute regulariztion gradient
    reg_grad = lamb * parameters

    return pred_grad + reg_grad


def check_grad(parameters, y, r, shape1, shape2, lamb=0.0):
    """
    Numeric gradient checking

    :param parameters: the flatten parameters.
    :param y: the ratings of each user to each movie, shape (n_m, n_u)
    :param r: whether or not a specific user rated a specific movie, shape (n_m, n_u)
    :param shape1: shape of the movies parameters matrix
    :param shape2: shape of the users parameters matrix
    :param lamb: optional,  regularization parameter lambda, scalar. default = 0.0
    :type parameters: np.ndarray
    :type y: np.ndarray
    :type r: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :return: None
    """
    eps = 1e-4
    analytic_grad = compute_grad(parameters, y, r, shape1, shape2, lamb)

    num_examples = 50  # checking all parameters takes too long, so a subset is checked.
    loc_vec = np.random.randint(0, len(parameters), num_examples)
    mean_error = 0.0

    for loc in loc_vec:
        param_p = np.copy(parameters)
        param_m = np.copy(parameters)

        param_p[loc] += eps
        param_m[loc] -= eps

        cost_p = compute_cost(param_p, y, r, shape1, shape2, lamb)
        cost_m = compute_cost(param_m, y, r, shape1, shape2, lamb)

        numeric_grad = (cost_p - cost_m) / (2 * eps)

        diff = np.abs(numeric_grad - analytic_grad[loc])
        mean_error += diff / num_examples

    print(mean_error)


def train(init_parameters, y, r, shape1, shape2, lamb=0.0):
    """
    Numeric gradient checking

    :param init_parameters: the initial flatten parameters.
    :param y: the ratings of each user to each movie, shape (n_m, n_u)
    :param r: whether or not a specific user rated a specific movie, shape (n_m, n_u)
    :param shape1: shape of the movies parameters matrix
    :param shape2: shape of the users parameters matrix
    :param lamb: optional,  regularization parameter lambda, scalar. default = 0.0
    :type init_parameters: np.ndarray
    :type y: np.ndarray
    :type r: np.ndarray
    :type shape1: (int, int)
    :type shape2: (int, int)
    :type lamb: float
    :return: learned parameter, shape like init_parameters
    :rtype: np.ndarray
    """
    op_result = minimize(compute_cost, init_parameters, (y, r, shape1, shape2, lamb),
                         jac=compute_grad, method='L-BFGS-B')

    if not op_result.success:
        print('something whent worng!!!')
        raise RuntimeError(op_result.message)

    return op_result.x


def get_movie_names():
    """
    get a list of movie names from '../data/movie_ids.txt'

    :return: list of movie names
    :rtype: list
    """
    fp = open('../data/movie_ids.txt')
    content = fp.readlines()
    content = [x.strip().split(" ", 1)[-1] for x in content]
    fp.close()
    return content
