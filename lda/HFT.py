import numpy as np
from datetime import datetime as dt
from scipy.optimize import minimize
import sys

from RatingModel import RatingModel
from ReviewModel import ReviewModel


class HFT:
    def __init__(self, ratings_filename='../Data/ratings.npz', reviews_filename='../Data/reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.rating_model.get_predicted_ratings()

        self.kappa = 1.0 # np.random.uniform()

        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.get_theta()
        self.review_model.Gibbsampler()

        self.mu = 1.0

        self.opt_iter = 0
        self.step_size = 0.001

    def get_theta(self):
        self.review_model.theta = np.exp(self.kappa * self.rating_model.gamma_item)
        partition = np.sum(self.review_model.theta, axis=1)
        self.review_model.theta /= partition[:, None]

    def flatten(self):
        return np.concatenate((np.array([self.rating_model.alpha]),
                               self.rating_model.beta_user,
                               self.rating_model.beta_item,
                               self.rating_model.gamma_user.flatten(),
                               self.rating_model.gamma_item.flatten(),
                               self.review_model.phi.flatten(),
                               np.array([self.kappa])))

    def structure(self, array):
        param_list = list()
        offset = 0

        # alpha
        param_list += [array[0]]
        offset += 1

        # beta_user
        ptr = offset + self.rating_model.n_users
        param_list += [array[offset:ptr]]
        offset = ptr

        # beta_item
        ptr = offset + self.rating_model.n_items
        param_list += [array[offset:ptr]]
        offset = ptr

        # gamma_user
        ptr = offset + self.rating_model.n_users*self.rating_model.n_hidden_factors
        param_list += [array[offset:ptr].reshape(self.rating_model.n_users, self.rating_model.n_hidden_factors)]
        offset = ptr

        # gamma_item
        ptr = offset + self.rating_model.n_items*self.rating_model.n_hidden_factors
        param_list += [array[offset:ptr].reshape(self.rating_model.n_items, self.rating_model.n_hidden_factors)]
        offset = ptr

        # phi
        ptr = offset + self.review_model.n_topics*self.review_model.n_vocab
        param_list += [array[offset:ptr].reshape(self.review_model.n_topics, self.review_model.n_vocab)]
        offset = ptr

        # kappa
        param_list += [array[offset]]

        return tuple(param_list)

    def get_gradients(self):
        self.get_theta()
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

    def get_error(self):
        return self.rating_model.get_rating_error() - self.mu * self.review_model.loglikelihood()

    """
        Performs gradient update using self.step_size
        Calls get_gradients to get gradients and performs update

        NOTE: After gradient update converges, call get_theta to recalculate theta values and then
                call GibbSampler to resample; call get_theta first to update theta used for sampling
            Also, only exponentiate and normalize phi after gradient update converges
    """
    def gradient_update(self):
        start_time = dt.now()
        self.opt_iter += 1
        gradients = self.get_gradients()
        self.rating_model.alpha -= self.step_size * gradients[0]
        self.rating_model.beta_user -= self.step_size * gradients[1]
        self.rating_model.beta_item -= self.step_size * gradients[2]
        self.rating_model.gamma_user -= self.step_size * gradients[3]
        self.rating_model.gamma_item -= self.step_size * gradients[4]
        self.review_model.phi -= self.step_size * gradients[5]
        #self.kappa -= self.step_size * gradients[6]
        print self.opt_iter, (dt.now() - start_time).seconds,[grad.max() for grad in gradients]

    def error_gradients(self, params):
        self.opt_iter += 1
        start_time = dt.now()

        alpha, beta_user, beta_item, gamma_user, gamma_item, phi, kappa = self.structure(params)

        # get rating error
        predicted_rating = np.zeros((self.rating_model.n_users, self.rating_model.n_items))
        ix = self.rating_model.data.nonzero()
        for u, i in zip(ix[0], ix[1]):
            predicted_rating[u, i] = np.dot(gamma_user[u, :], gamma_item[i, :]) + alpha + beta_user[u] + beta_item[i]

        rating_loss = predicted_rating - self.rating_model.data
        rating_error = np.sum(np.square(rating_loss))
        rating_error_time = (dt.now() - start_time).seconds

        # get log_likelihood
        theta = np.exp(kappa * gamma_item)
        theta /= np.sum(theta, axis=1)[:, None]

        log_likelihood = 0
        for i in xrange(self.review_model.n_docs):
            words = self.review_model.reviews[i]
            topics = self.review_model.z[i]
            log_likelihood += np.sum(np.log(theta[i, topics]) + np.log(phi[topics, words]))
        review_error_time = (dt.now() - start_time).seconds - rating_error_time

        error = rating_error - self.mu * log_likelihood

        # get gradients
        review_loss = (self.review_model.topic_frequencies -
                       theta * np.sum(self.review_model.topic_frequencies, axis=1)[:, None])

        alpha_gradient = 2 * np.sum(rating_loss)
        beta_item_gradients = 2 * np.ravel(np.sum(rating_loss, axis=0))
        beta_user_gradients = 2 * np.ravel(np.sum(rating_loss, axis=1))
        gamma_user_gradients = 2 * np.asarray(np.dot(rating_loss, gamma_item))
        gamma_item_gradients = 2 * np.asarray(np.dot(rating_loss.transpose(), gamma_user)) - self.mu*kappa*review_loss
        phi_gradients = np.divide(self.review_model.word_topic_frequencies, phi)
        kappa_gradient = np.sum(gamma_item * review_loss)

        # try:
        gradients = np.concatenate((np.array([alpha_gradient]), beta_user_gradients, beta_item_gradients,
                                   gamma_user_gradients.flatten(), gamma_item_gradients.flatten(),
                                   phi_gradients.flatten(), np.array([kappa_gradient])))
        #     gradients = np.concatenate((np.array([alpha_gradient]),
        #                                 beta_user_gradients,
        #                                 beta_item_gradients,
        #                                 gamma_user_gradients.flatten(),
        #                                 gamma_item_gradients.flatten(),
        #                                 phi_gradients.flatten(),
        #                                 np.array([kappa_gradient])))
        # except:
        #     print np.array([alpha_gradient]).shape
        #     print beta_user_gradients.shape
        #     print beta_item_gradients.shape
        #     print gamma_user_gradients.shape, gamma_user_gradients.flatten().shape
        #     print gamma_item_gradients.shape, gamma_item_gradients.flatten().shape
        #     print phi_gradients.shape, phi_gradients.flatten().shape
        #     print np.array([kappa_gradient]).shape
        #     sys.exit(1)

        print self.opt_iter, rating_error_time, review_error_time, (dt.now() - start_time).seconds, start_time.time()
        return error, gradients

    def update(self):
        params = self.flatten()

        self.opt_iter = 0
        opt_params = minimize(self.error_gradients, params, method='L-BFGS-B', jac=True).x
        self.rating_model.alpha, self.rating_model.beta_user, self.rating_model.beta_item, \
            self.rating_model.gamma_user, self.rating_model.gamma_item, \
            self.review_model.phi, self.kappa = self.structure(opt_params)
        self.review_model.phi = np.exp(self.review_model.phi)
        self.review_model.phi /= self.review_model.phi.sum(axis=1)[:, None]

if __name__ == '__main__':
    print sys.version
    print 'Running main method...'

    start_time = dt.now()
    hft = HFT(ratings_filename='Data/ratings.npz', reviews_filename='Data/reviews.npz')
    print 'Finished loading model in', (dt.now() - start_time).seconds, 'seconds'

    #start_time = dt.now()
    #grads = hft.get_gradients()
    #print 'Finished calculating gradients in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    hft.rating_model.get_predicted_ratings()
    print 'Finished predicting new ratings in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    hft.review_model.Gibbsampler()
    print 'Finished performing Gibbs sampling in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    l = hft.review_model.loglikelihood()
    print 'Finished calculating log-likelihood in', (dt.now() - start_time).seconds, 'seconds'

    start_time = dt.now()
    # hft.update()
    #print 'Finished updating parameters in', (dt.now() - start_time).seconds, 'seconds over', hft.bfgs_iter, 'iterations'

    for i in xrange(100):
        hft.gradient_update()
    print 'Finished updating parameters in', (dt.now() - start_time).seconds, 'seconds'

