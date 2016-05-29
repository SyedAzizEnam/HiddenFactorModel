from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd
import cPickle as pickle

review_fileName = '../Data/bow_reviews.csv'
vocab_fileName = '../Data/vocab.txt'
n_reviews = 1363012
n_words = 13249


class MatrixConstructor:

    def __init__(self):
        self.review_data = pd.read_csv(review_fileName)
        print 'Finished loading data...'

        # Mapping all business_ids
        self.business_ids = list()
        for business in self.review_data['business_id']:
            self.business_ids.append(business)
        unique_business_ids = list(set(self.business_ids))
        self.n_businesses = len(unique_business_ids)

        self.business_dict = dict()
        for index, b_id in enumerate(unique_business_ids):
            self.business_dict[index] = b_id
            self.business_dict[b_id] = index

        # Mapping all user_ids
        self.user_ids = list()
        for user in self.review_data['user_id']:
            self.user_ids.append(user)
        unique_user_ids = list(set(self.user_ids))
        self.n_users = len(unique_user_ids)

        self.user_dict = dict()
        for index, u_id in enumerate(unique_user_ids):
            self.user_dict[index] = u_id
            self.user_dict[u_id] = index

        self.reviews = lil_matrix((self.n_businesses, n_words))
        self.ratings = lil_matrix((self.n_users, self.n_businesses))

    def get_reviews(self):
        for r_ix, review in enumerate(self.review_data['text']):
            b_id = self.business_ids[r_ix]
            b_ix = self.business_dict[b_id]

            words = review.split()
            for word in words:
                word = word.split(':')
                self.reviews[b_ix, int(word[0])] += int(word[1])

    def get_ratings(self):
        rating_counts = lil_matrix(self.ratings.shape)
        for r_ix, rating in enumerate(self.review_data['stars']):
            b_id = self.business_ids[r_ix]
            b_ix = self.business_dict[b_id]

            u_id = self.user_ids[r_ix]
            u_ix = self.user_dict[u_id]

            if rating_counts[u_ix, b_ix] == 0:
                rating_counts[u_ix, b_ix] = 1
                self.ratings[u_ix, b_ix] = rating
            else:
                prior_rating = self.ratings[u_ix, b_ix]
                prior_count = rating_counts[u_ix, b_ix]
                new_count = prior_count+1
                new_rating = (prior_rating*prior_count + rating)/new_count

                rating_counts[u_ix, b_ix] = new_count
                self.ratings[u_ix, b_ix] = new_rating

    def save_reviews(self, review_output_filename='../Data/reviews.npz'):
        self.reviews = csr_matrix(self.reviews)
        np.savez(open(review_output_filename, 'wb'),
                 data=self.reviews.data, indices=self.reviews.indices, indptr=self.reviews.indptr,
                 shape=self.reviews.shape)

    def save_ratings(self, ratings_output_filename='../Data/ratings.npz'):
        self.ratings = csr_matrix(self.ratings)
        np.savez(open(ratings_output_filename, 'wb'),
                 data=self.ratings.data, indices=self.ratings.indices, indptr=self.ratings.indptr,
                 shape=self.ratings.shape)

    def save_ids(self, business_output_filename='../Data/business_ids.pkl',
                 user_output_filename='../Data/user_ids.pkl'):
        pickle.dump(self.business_dict, open(business_output_filename, 'wb'))
        pickle.dump(self.user_dict, open(user_output_filename, 'wb'))

if __name__ == '__main__':
    print 'Loading data into memory...'
    mat_constr = MatrixConstructor()

    print 'Constructing review matrix...'
    mat_constr.get_reviews()
    print 'Saving review matrix...'
    mat_constr.save_reviews()

    print 'Constructing ratings dictionary...'
    mat_constr.get_ratings()
    print 'Saving ratings dictionary...'
    mat_constr.save_ratings()

    print 'Saving id structures...'
    mat_constr.save_ids()

    print 'Processing complete!'
