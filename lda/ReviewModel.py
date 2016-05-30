import numpy as np
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
        self.phi = np.zeros((n_topics, self.n_vocab))
        self.theta = np.zeros((self.n_docs, n_topics))

        self.z = list()
        self.reviews = list()
        for doc_ix in xrange(self.n_docs):
            data_review = data[doc_ix, :].toarray()[0]
            n_words = int(np.sum(data_review))
            self.z.append(np.zeros(n_words, dtype=int))
            self.reviews.append(flatten_bow(data_review))
