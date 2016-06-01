import numpy as np

from RatingModel import RatingModel
from ReviewModel import ReviewModel


class HFT:

    def __init__(self, ratings_filename='../Data/ratings.npz', reviews_filename='../Data/reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.kappa = 0.0

    def get_theta(self):
        self.review_model.theta = np.exp(self.kappa * self.rating_model.gamma_item)
        partition = np.sum(self.review_model.theta, axis=1)
        self.review_model.theta /= partition[:, None]

    def get_word_topic_frequencies(self):
        frequencies = np.ndarray((self.review_model.n_topics, self.review_model.n_vocab), dtype=int)
        for review, topics in zip(self.review_model.reviews, self.review_model.z):
            for word, topic in zip(review, topics):
                frequencies[topic, word] += 1
        return frequencies

    def get_beta_item_gradients(self):
        return 2 * np.sum(self.rating_model.predicted_rating - self.rating_model.data, axis=0)

    def get_beta_user_gradients(self):
        return 2 * np.sum(self.rating_model.predicted_rating - self.rating_model.data, axis=1)

    def get_gamma_user_gradients(self):
        return 2 * np.dot((self.rating_model.predicted_rating - self.rating_model.data),
                          self.rating_model.gamma_item)

    def get_kappa_gradient(self):
        kappa_gradient = 0
        for b_ix in xrange(self.review_model.n_docs):
            topics = self.review_model.z[b_ix]
            kappa_gradient += np.sum(self.rating_model.gamma_item[b_ix, topics] *
                                     (1 - self.review_model.theta[b_ix, topics]))
            return kappa_gradient

    def get_alpha_gradient(self):
        return 2*np.sum(self.rating_model.predicted_rating - self.rating_model.data)

    def get_phi_gradient(self):
        return np.divide(self.get_word_topic_frequencies(), self.review_model.phi)
