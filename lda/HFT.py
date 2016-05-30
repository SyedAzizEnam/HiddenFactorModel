import numpy as np

from RatingModel import RatingModel
from ReviewModel import ReviewModel


class HFT:

    def __init__(self, ratings_filename='../Data/ratings.npz', reviews_filename='../Data/reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.kappa = 0.0

    def get_theta(self):
        self.review_model.theta = np.exp(self.kappa*self.rating_model.gamma_item)
        partition = np.sum(self.review_model.theta, axis=1)
        self.review_model.theta /= partition[:, None]

    def get_word_topic_frequencies(self):
        frequencies = np.ndarray((self.review_model.n_topics, self.review_model.n_vocab), dtype=int)
        for review, topics in zip(self.review_model.reviews, self.review_model.z):
            for word, topic in zip(review, topics):
                frequencies[topic, word] += 1
        return frequencies
