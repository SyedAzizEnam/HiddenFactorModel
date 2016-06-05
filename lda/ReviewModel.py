import numpy as np
from scipy.sparse import csr_matrix
import random
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


class ReviewModel:
    def __init__(self, reviews_filename='../Data/reviews.npz', n_topics=10):
        data = DataLoader.review_data(reviews_filename)
        self.n_docs, self.n_vocab = data.shape
        self.n_topics = n_topics

        self.phi = np.random.rand(n_topics, self.n_vocab)
        self.phi = self.phi/self.phi.sum(axis=1)[:, None]

        self.theta = np.random.rand(self.n_docs, n_topics)
        self.theta = self.theta/self.theta.sum(axis=1)[:, None]

        self.topic_frequencies = np.zeros((self.n_docs, self.n_topics))
        self.word_topic_frequencies = np.zeros((self.n_vocab, self.n_topics))

        self.z = list()
        self.reviews = list()
        for doc_ix in xrange(self.n_docs):
            data_review = data[doc_ix, :].toarray()[0]
            n_words = int(np.sum(data_review))
            self.z.append(np.zeros(n_words, dtype=int))
            self.reviews.append(flatten_bow(data_review))

    def loglikelihood(self):
        """Computes likelihood of a corpus

        Returns
        -------
        loglikelihood: The loglikelihood of the entire corpus
        """

        All_loglikelihoods = list()

        for i in xrange(self.n_docs):
            
            words = self.reviews[i]
            topics = self.z[i]
            loglikelihood = 0

            loglikelihood = np.log(self.theta[([i]*len(topics),topics)]) + \
                              np.log(self.phi[(topics,words)])

            All_loglikelihoods.append(np.sum(loglikelihood))

        return sum(All_loglikelihoods)

    def Gibbsampler(self):
        """
        Resamples the topic_assingments accross the entires corpus

        Returns: 
        new_topic_assingments: list of numpy arrays
        """

        new_topic_assingments = list()

        self.topic_frequencies = np.zeros((self.n_docs, self.n_topics))
        self.word_topic_frequencies = np.zeros((self.n_vocab, self.n_topics))
        
        for i in xrange(self.n_docs):
            
            words = self.reviews[i]
            topic_assingments = []
            
            for word in words:
                p = self.theta[i, :]*self.phi[:, word]
                p = p/p.sum()
                new_topic = self.sampleWithDistribution(p)
                topic_assingments.append(new_topic)

                self.topic_frequencies[i,new_topic] += 1.0
                self.word_topic_frequencies[word, new_topic] += 1.0
            
            topic_assingments = np.array(topic_assingments)
            new_topic_assingments.append(topic_assingments)

        self.z = new_topic_assingments

    def sampleWithDistribution(self, p):
        """ Sampler that samples with respect to distribution p

        Parameters
        ----------
        p : numpy array

        Returns
        -------
        i: index of sampled value
        """
        r = random.random()  # Rational number between 0 and 1
        
        for i in xrange(len(p)):
            r = r - p[i]
            if r <= 0:
                return i
        raise Exception("Uh Oh... selectWithDistribution with r value %f" %r)