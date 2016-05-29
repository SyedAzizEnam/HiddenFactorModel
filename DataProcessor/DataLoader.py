import numpy as np
from scipy.sparse import csr_matrix
import cPickle as pickle

review_data_filename = '../Data/reviews.npz'
ratings_data_filename = '../Data/ratings.pkl'
user_id_filename = '../Data/user_ids.pkl'
business_id_filename = '../Data/business_ids.pkl'


def review_data():
    loader = np.load(review_data_filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def ratings_data():
    return pickle.load(open(ratings_data_filename, 'rb'))


def user_ids():
    return pickle.load(open(user_id_filename, 'rb'))


def business_ids():
    return pickle.load(open(business_id_filename, 'rb'))
