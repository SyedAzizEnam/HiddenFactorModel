import numpy as np
from DataProcessor import DataLoader


class ReviewModel:

    def __init__(self, reviews_filename='../Data/reviews.npz', n_topics=10):
        self.data = DataLoader.review_data(reviews_filename)

        n_docs, n_vocab = self.data.shape
        self.topic_model = np.zeros((n_topics, n_vocab))
        self.topic_proportion = np.zeros((n_docs, n_topics))

        self.topic_assignment = list()
        for doc_ix in xrange(n_docs):
            n_words = int(np.sum(self.data[doc_ix, :].toarray()))
            self.topic_assignment.append(np.zeros(n_words, dtype=int))
