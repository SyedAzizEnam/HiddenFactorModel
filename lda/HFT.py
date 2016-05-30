from RatingModel import RatingModel
from ReviewModel import ReviewModel


class HFT:

    def __init__(self, ratings_filename='../Data/ratings.npz', reviews_filename='../Data/reviews.npz', n_hidden=10):
        self.rating_model = RatingModel(ratings_filename, n_hidden)
        self.review_model = ReviewModel(reviews_filename, n_hidden)
        self.kappa = 0.0
