import numpy as np
from scipy.sparse import csr_matrix


def review_data():
    loader = np.load('../Data/reviews.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def ratings_data():
    return np.load('../Data/ratings.npy')
