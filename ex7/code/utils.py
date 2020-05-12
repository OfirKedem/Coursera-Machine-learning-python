import numpy as np
import matplotlib.pyplot as plt


# NOTE:
# m = # of data points
# n = # of features
# k = # of clusters


# 1 K-means Clustering

def find_closest_centroids(x, centroids):
    """
    find closest centroid for each data point. no loops

    :param x: data, shape (m,n)
    :param centroids: the coordinate of the centroids, shape (K, n)
    :type x: np.ndarray
    :type centroids: np.ndarray
    :return: assigment to each data-point to it's closest centroid. each entry contains the index of the centroid.
     shape (m,)
    :rtype: np.ndarray
    """
    k = centroids.shape[0]
    x_rep = np.repeat(x[:, :, np.newaxis], k, axis=2)  # (m,n,K)
    dist = ((x_rep - centroids.T) ** 2).sum(axis=1)  # (m,K)
    c = dist.argmin(axis=1)
    return c


def compute_centroids(x, c, k):
    """
    computes the new centroids, by the center of each group of points.

    :param x: data, shape(m,n)
    :param c: assigment vector, shape(m,)
    :param k: number of centroids, scalar
    :type x: np.ndarray
    :type c: np.ndarray
    :type k: int
    :return: the new centroids, shape(k,n)
    :rtype: np.ndarray
    """
    n = x.shape[1]
    centroids = np.zeros([k, n])

    for i in range(k):
        c_idx = c == i
        c_idx_size = c_idx.sum()
        centroids[i] = x[c_idx, :].sum(axis=0) / c_idx_size
    return centroids


def run_k_means(x, initial_centroids, max_iters=10):
    """
    runs the K-means algo. NO stop condition, runs for all iterations

    :param x: data, shape(m,n)
    :param initial_centroids: initial coordinates of centroids, shape(K,n)
    :param max_iters: number of iteration, scalar
    :type x: np.ndarray
    :type initial_centroids: np.ndarray
    :type max_iters: int
    :return: the assigment of each data-point to the closest centroid. shape(m,)
     the centroids, shape(k,n)
    :rtype (np.ndarray, np.ndarray)

    """
    k = initial_centroids.shape[0]
    c = np.zeros_like(initial_centroids)
    centroids = initial_centroids
    for i in range(max_iters):
        c = find_closest_centroids(x, centroids)
        centroids = compute_centroids(x, c, k)
    return c, centroids


# 1.2 K-means on example dataset
def plot_k_means(x, c, centroids):
    """
    plots the result of K-means

    :param x: data, shape(m,n)
    :param c: assigment of each data-point to a centroid. shape(m,)
    :param centroids: the centroids coordinate, shape(K,n)
    :type x: np.ndarray
    :type c: np.ndarray
    :type centroids: np.ndarray
    :return: None
    """
    plt.scatter(x[:, 0], x[:, 1], c=c)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='x')


# 1.3 Random initialization
def k_means_init_centroids(x, k):
    """ picks k random data-points from x without repetition.

    :param x: data, shape(m,n)
    :param k: number of centroids, scalar
    :type x: np.ndarray
    :type k: int
    :return: K random data-points from x. shape(k,n)
    :rtype: np.ndarray
    """
    rand_idx = np.random.permutation(x.shape[0])[:k]
    return x[rand_idx]


# 2 Principal Component Analysis

# 2.2 Implementing PCA
def feature_normalize(x):
    """
    mean center and feature scaling

    :param x: data to normalize, shape (m,_)
    :type x: np.ndarray
    :return: normalize data, shape (same as x)
    :rtype: np.ndarray
    """
    return (x - x.mean(axis=0)) / x.std(axis=0)


def pca(x):
    """
    perform PCA on X.

    :param x: normalized data , shape(m,n)
    :type x: np.ndarray
    :returns:
        u: principal components in the columns, shape(n,n).
        s:  singular values, shape(n,)
    :rtype: (np.ndarray, np.ndarray)
    """
    m = x.shape[0]
    cov = x.T.dot(x) / m

    # NOTE: the commented lines below are all equivalent ways to compute pca.

    # u,s,vh = np.linalg.svd(X)
    # return vh.T,s

    # w,v = np.linalg.eig(covM)
    # return v, w

    u, s, vh = np.linalg.svd(cov)
    return u, s


# 2.3 Dimensionality Reduction with PCA
def project_data(x, u, k):
    """
    project the data on to the first K principal components.

    :param x: data, shape(m,n)
    :param u: principal components in the columns. shape(n,n)
    :param k: dim to reduce to, scalar
    :type x: np.ndarray
    :type u: np.ndarray
    :type k: int
    :return: the data projected on to the k principal components span, shape(m,K)
    :rtype: np.ndarray
    """

    projector = u[:, :k]  # (n,K)
    return x.dot(projector)  # (m,K)


# 2.3.2 Reconstructing an approximation of the data
def recover_data(z, u, k):
    """
    recovers data that was compressed by PCA with the k first components of u

    :param z: projected data, shape(m,K)
    :param u: principal components in the columns. shape(n,n)
    :param k: the dim of the projected data, scalar
    :type z: np.ndarray
    :type u: np.ndarray
    :type k: int
    :return: recovered data, shape(m,n)
    :rtype: np.ndarray
    """

    recover_mat = u[:, :k].T
    x_rec = z.dot(recover_mat)
    return x_rec

