import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from datetime import datetime

from utils import Utils
from config import Config

config = Config()
utils = Utils()


def collaborative_filtering(ratings_new):
    # Collaborative Filtering Based Recommendation Systems

    # To cope up with computing power I have and to reduce the data set size, I am considering users
    # who have rated at least 100 books and books which have at least 100 ratings
    counts1 = ratings_new['user'].value_counts()
    ratings_new = ratings_new[ratings_new['user'].isin(counts1[counts1 >= config.cf_constant].index)]
    counts = ratings_new['bookRating'].value_counts()
    ratings_new = ratings_new[ratings_new['bookRating'].isin(counts[counts >= config.cf_constant].index)]

    # Generating ratings matrix from explicit ratings table
    ratings_matrix = ratings_new.pivot(index='user', columns='bookId', values='bookRating')

    # since NaNs cannot be handled by training algorithms, replacing these by 0,
    # which indicates absence of ratings and set data type as integer
    ratings_matrix.fillna(0, inplace=True)
    ratings_matrix = ratings_matrix.astype(np.int32)

    return ratings_matrix


def preprocess():
    books = pd.read_csv(config.book_csv, sep=',', error_bad_lines=False, encoding="latin-1")
    users = pd.read_csv(config.user_csv, sep=',', error_bad_lines=False, encoding="latin-1")
    ratings = pd.read_csv(config.user_event_csv, sep=',', error_bad_lines=False, encoding="latin-1")

    """     Cleaning the books dataframe        """
    books = utils.clean_book_dataframe(books)

    # """     Cleaning User df        """
    # users = utils.clean_user_dataframe(users)

    """     Cleaning user event data frame and mapping the impression to rating     """
    ratings = utils.clean_events_dataframe(ratings)
    # ratings data set should have books only which exist in our books data set,
    #  unless new books are added to books data set
    ratings_new = ratings[ratings.bookId.isin(books.bookISBN)]

    """     Further analysis        """
    n_users = users.shape[0]
    n_books = books.shape[0]
    print("Number of users : {}".format(n_users))
    print("Number of books : {}".format(n_books))

    ratings_matrix = collaborative_filtering(ratings_new)
    return ratings_matrix
