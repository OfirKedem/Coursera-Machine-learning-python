import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def plot_decision_boundary(clf, x, y, res=1000, extra_b=0.5):
    """
     plots the decision boundary of the classifier clf. plots the data set as well.

    :param clf: classifier
    :param x: data, shape (m,2)
    :param y: labels, shape(m, )
    :param res:  resolution of boundary
    :param extra_b: by how much to extend the range of the boundary
    :type clf: svm.SVC
    :type x: np.ndarray
    :type y: np.ndarray
    :type res: int
    :type extra_b: float
    :return: None
    """
    x_range = np.linspace(x[:, 0].min() - extra_b, x[:, 0].max() + extra_b, res)
    y_range = np.linspace(x[:, 1].min() - extra_b, x[:, 1].max() + extra_b, res)

    xx, yy = np.meshgrid(x_range, y_range)

    x_plot = np.stack([xx.ravel(), yy.ravel()], axis=1)
    z = clf.predict(x_plot).reshape(xx.shape)

    plt.contour(xx, yy, z, levels=[0.5], colors='blue')
    plt.scatter(x[:, 0], x[:, 1], c=y)


# 1.2 SVM with Gaussian Kernels

# 1.2.1 Gaussian Kernel
def gaussian_kernel(x, y):
    """
    compute the gaussian kernel between to vectors

    :param x: vectors in the rows
    :param y: vectors in the rows
    :type x: np.ndarray
    :type y: np.ndarray
    :return: gaussian kernel between any vector in x and y
    :rtype: np.ndarray

    .. note:: the predict method of sklearn classifier calls this function with different size matrices which need to be
    computed in a loop so this is a very slow implementation and i don't use it.
    """
    sigma = 1
    print('x: ' + str(x.shape))
    print('y: ' + str(y.shape))
    n_samples_x = x.shape[0]
    n_samples_y = y.shape[0]

    if n_samples_x == n_samples_y:
        return np.exp(-(x - y).dot((x - y).T) / (2 * sigma ** 2))

    kernel = np.zeros([n_samples_x, n_samples_y])
    for i in range(n_samples_x):
        for j in range(n_samples_y):
            kernel[i, j] = np.exp(-(x[i] - y[j]).dot((x[i] - y[j]).T) / (2 * sigma ** 2))
    return kernel
