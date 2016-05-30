import numpy as np
from DataProcessor import DataLoader


class ReviewModel:

    def __init__(self, reviews_filename='../Data/reviews.npz', n_topics=10):
        self.data = DataLoader.review_data(reviews_filename)

        n_docs, n_vocab = self.data.shape
        self.phi = np.zeros((n_topics, n_vocab))
        self.theta = np.zeros((n_docs, n_topics))

        self.z = list()
        for doc_ix in xrange(n_docs):
            n_words = int(np.sum(self.data[doc_ix, :].toarray()))
            self.z.append(np.zeros(n_words, dtype=int))
