from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd
import sys


class MatrixConstructor:
    def __init__(self):
        self.review_fileName = '../Data/bow_reviews.csv'
        self.vocab_fileName = '../Data/vocab.txt'
        self.n_docs = 1363012
        self.n_words = 13249
        self.review_data = lil_matrix((self.n_docs, self.n_words))
        self.rating_data = np.ndarray([self.n_docs])

    def construct_matrix(self):
        reviews = pd.read_csv(self.review_fileName)
        for data in reviews.iterrows():
            index = data[0]
            rating = data[1]['stars']
            review = data[1]['text'].split()
            for word in review:
                word = word.split(':')
                try:
                    self.review_data[index, int(word[0])] += int(word[1])
                    self.rating_data[index] = rating
                except:
                    print index, word
                    sys.exit(1)

    def save_review_matrix(self, output_filename):
        self.review_data = csr_matrix(self.review_data)
        np.savez(open(output_filename, 'wb'),
                 data=self.review_data.data,
                 indices=self.review_data.indices, indptr=self.review_data.indptr, shape=self.review_data.shape)

    def save_ratings_matrix(self, output_filename):
        np.save(open(output_filename, 'wb'), self.rating_data)


if __name__ == '__main__':
    print 'Loading data...'
    mat_constr = MatrixConstructor()
    print 'Finished loading!'

    print 'Constructing matrices...'
    mat_constr.construct_matrix()

    print 'Saving review matrix...'
    mat_constr.save_review_matrix('../Data/reviews.npz')
    print 'Processing review matrix complete!'

    print 'Saving ratings matrix...'
    mat_constr.save_ratings_matrix('../Data/ratings.npy')
    print 'Processing ratings matrix complete!'

    print 'All data processed.'
