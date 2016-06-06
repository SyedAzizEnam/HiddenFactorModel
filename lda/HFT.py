import numpy as np
from datetime import datetime as dt

from RatingModel import RatingModel
from ReviewModel import ReviewModel


class HFT:
    def __init__(self, ratings_filename='../Data/ratings.npz', reviews_filename='../Data/reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.rating_model.get_predicted_ratings()

        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.review_model.Gibbsampler()

        self.kappa = 0.0
        self.mu = 1.0

    def get_theta(self):
        self.review_model.theta = np.exp(self.kappa * self.rating_model.gamma_item)
        partition = np.sum(self.review_model.theta, axis=1)
        self.review_model.theta /= partition[:, None]

    # def get_word_topic_frequencies(self):
    #     frequencies = np.ndarray((self.review_model.n_topics, self.review_model.n_vocab), dtype=int)
    #     for review, topics in zip(self.review_model.reviews, self.review_model.z):
    #         for word, topic in zip(review, topics):
    #             frequencies[topic, word] += 1
    #     return frequencies

    # def get_beta_item_gradients(self, rating_loss):
    #     return 2 * np.sum(rating_loss, axis=0)
    #
    # def get_beta_user_gradients(self, rating_loss):
    #     return 2 * np.sum(rating_loss, axis=1)
    #
    # def get_gamma_user_gradients(self, rating_loss):
    #     return 2 * np.dot(rating_loss, self.rating_model.gamma_item)
    #
    # def get_gamma_item_gradients(self, rating_loss, review_loss):
    #     return 2 * np.dot(rating_loss,
    #                       self.rating_model.gamma_item) - self.mu * self.kappa * review_loss
    #
    # def get_alpha_gradient(self, rating_loss):
    #     return 2 * np.sum(rating_loss)
    #
    # def get_phi_gradients(self):
    #     return np.divide(self.get_word_topic_frequencies(), self.review_model.phi)
    #
    # def get_kappa_gradient(self, review_loss):
    #     return np.sum(self.rating_model.gamma_item * review_loss)

    def get_gradients(self):
        rating_loss = self.rating_model.predicted_rating - self.rating_model.data
        review_loss = (self.review_model.topic_frequencies -
                       self.review_model.theta * np.sum(self.review_model.topic_frequencies, axis=1)[:, None])

        alpha_gradient = 2 * np.sum(rating_loss)
        beta_item_gradients = 2 * np.ravel(np.sum(rating_loss, axis=0))
        beta_user_gradients = 2 * np.ravel(np.sum(rating_loss, axis=1))
        gamma_user_gradients = 2 * np.dot(rating_loss, self.rating_model.gamma_item)
        gamma_item_gradients = 2 * np.dot(rating_loss.transpose(), self.rating_model.gamma_user) - \
                                    self.mu*self.kappa*review_loss
        phi_gradients = np.divide(self.review_model.word_topic_frequencies, self.review_model.phi)
        kappa_gradient = np.sum(self.rating_model.gamma_item * review_loss)

        return [alpha_gradient, beta_user_gradients, beta_item_gradients,
                gamma_user_gradients, gamma_item_gradients, phi_gradients, kappa_gradient]

if __name__ == '__main__':
    print 'Running main method...'

    start_time = dt.now()
    hft = HFT(ratings_filename = 'Data/ratings.npz', reviews_filename = 'Data/reviews.npz')
    print 'Finished loading model in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    grads = hft.get_gradients()
    print 'Finished calculating gradients in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    hft.rating_model.get_predicted_ratings()
    print 'Finished predicting new ratings in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    hft.review_model.Gibbsampler()
    print 'Finished performing Gibbs sampling in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    l = hft.review_model.loglikelihood()
    print 'Finished calculating log-likelihood in', (dt.now() - start_time).seconds, 'seconds'

