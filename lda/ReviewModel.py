import numpy as np
import sys
from DataProcessor import DataLoader


def flatten_bow(bow_review):
    """
    Flattens the bag of words array so that it is the
    same shape as the topic assingment array (z)

    Parameters
    ----------
    bow_review

    Returns
    -------
    words

    """
    indices = np.flatnonzero(bow_review)
    words = []
    for j in indices:
        words += [j]*int(bow_review[j])
    return np.array(words)


def sampleWithDistribution(p):
    """ Sampler that samples with respect to distribution p

    Parameters
    ----------
    p : numpy array

    Returns
    -------
    i: index of sampled value
    """
    r = np.random.uniform()  # Rational number between 0 and 1

    for i in xrange(len(p)):
        r = r - p[i]
        if r <= 0:
            return i
    raise Exception("Uh Oh... selectWithDistribution with r value %f" % r)


class ReviewModel:
    def __init__(self, reviews_filename='../Data/reviews.npz', n_topics=10):
        data = DataLoader.review_data(reviews_filename)
        self.n_docs, self.n_vocab = data.shape
        self.n_topics = n_topics

        self.phi = np.random.uniform(size=(n_topics, self.n_vocab))
        self.phi /= self.phi.sum(axis=1)[:, None]

        self.theta = np.random.uniform(size=(self.n_docs, n_topics))
        self.theta /= self.theta.sum(axis=1)[:, None]

        self.topic_frequencies = np.zeros((self.n_docs, self.n_topics))
        self.word_topic_frequencies = np.zeros((self.n_topics, self.n_vocab))

        self.z = list()
        self.reviews = list()
        self.backgroundwords = np.zeros(self.n_vocab)
<<<<<<< HEAD
        self.item_words = np.zeros(self.n_docs)

=======
        
>>>>>>> 5d1052cd28ad440fab568229de06e013fa7cf49b
        for doc_ix in xrange(self.n_docs):
            data_review = data[doc_ix, :].toarray()[0]
            n_words = int(np.sum(data_review))
            self.z.append(np.zeros(n_words, dtype=int))
            self.reviews.append(flatten_bow(data_review))
            self.item_words[doc_ix] = len(self.reviews[-1])
            for entry in self.reviews[-1]:
                self.backgroundwords[entry] += 1.0

        self.backgroundwords /= self.backgroundwords.sum()




            for entry in self.reviews[-1]:
                self.backgroundwords[entry] += 1.0

        self.backgroundwords /= self.backgroundwords.sum()


    def loglikelihood(self):
        """Computes likelihood of a corpus

        Returns
        -------
        loglikelihood: The loglikelihood of the entire corpus
        """
        log_likelihood = 0
        for i in xrange(self.n_docs):
            words = self.reviews[i]
            topics = self.z[i]
            log_likelihood += np.sum(np.log(self.theta[i, topics]) + np.log(self.phi[topics, words]))
        return log_likelihood

    def Gibbsampler(self):
        """
        Resamples the topic_assingments accross the entires corpus

        Returns: 
        new_topic_assingments: list of numpy arrays
        """

        new_topic_assignments = list()

        self.topic_frequencies.fill(0)
        self.word_topic_frequencies.fill(0)
        
        for i in xrange(self.n_docs):
            
            words = self.reviews[i]

            p = self.theta[i, :] * self.phi[:, words].transpose()
            p /= np.sum(p, axis=1)[:, None]
            p = p.tolist()

            topic_assignments = map(sampleWithDistribution, p)

            np.add.at(self.topic_frequencies[i], topic_assignments, 1)
            np.add.at(self.word_topic_frequencies, [topic_assignments, words], 1)

            new_topic_assignments.append(np.array(topic_assignments))

        self.z = new_topic_assignments
