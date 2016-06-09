import numpy as np
from DataProcessor import DataLoader


class RatingModel:

    def __init__(self, ratings_filename='../Data/ratings.npz', n_hidden_factors=10):
        self.data = DataLoader.ratings_data(ratings_filename)
        self.n_users, self.n_items = self.data.shape
        self.n_hidden_factors = n_hidden_factors
        self.corpus_ix = self.data.nonzero()

        self.alpha = np.random.uniform()
        self.beta_user = np.random.uniform(size=(self.n_users))
        self.beta_item = np.random.uniform(size=(self.n_items))
        self.gamma_user = np.random.uniform(size=(self.n_users, self.n_hidden_factors))
        self.gamma_item = np.random.uniform(size=(self.n_items, self.n_hidden_factors))
        self.predicted_rating = np.zeros((self.n_users, self.n_items))

    def get_predicted_ratings(self):
<<<<<<< HEAD
        ix = self.data.nonzero()
        for u, i in zip(ix[0], ix[1]):
            self.predicted_rating[u, i] = np.dot(self.gamma_user[u, :], self.gamma_item[i, :]) + \
                                          self.alpha + self.beta_user[u] + self.beta_item[i]
        # self.predicted_rating += self.alpha + self.beta_user[:, None] + self.beta_item
        # self.predicted_rating[np.logical_not(self.corpus_ix)] = 0
||||||| merged common ancestors
        ix = self.data.nonzero()
        for u, i in zip(ix[0], ix[1]):
            self.predicted_rating[u, i] = np.dot(self.gamma_user[u, :], self.gamma_item[i, :])
        self.predicted_rating += self.alpha + self.beta_user[:, None] + self.beta_item
        self.predicted_rating[np.logical_not(self.corpus_ix)] = 0
=======
        for u, i in zip(self.corpus_ix[0], self.corpus_ix[1]):
            self.predicted_rating[u, i] = np.dot(self.gamma_user[u, :], self.gamma_item[i, :]) + self.alpha + \
                                          self.beta_user[u] + self.beta_item[i]
        # self.predicted_rating += self.alpha + self.beta_user[:, None] + self.beta_item
        # self.predicted_rating[np.logical_not(self.corpus_ix)] = 0
>>>>>>> f15942e80518e0759bb7d1065584368076821e6c

    def get_rating_error(self):
        return np.sum(np.square(self.predicted_rating - self.data))
