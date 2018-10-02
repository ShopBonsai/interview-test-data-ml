import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

from preprocessing import preprocess
from utils import Utils
from config import Config

config = Config()
utils = Utils()


# This function finds k similar items given the item_id and ratings matrix

def findksimilaritems(item_id, ratings, metric='cosine', k=10):
    similarities = []
    indices = []
    ratings = ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    return similarities, indices


# This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(user_id, item_id, ratings, metric='cosine', k=10):
    prediction = wtd_sum = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices = findksimilaritems(item_id, ratings)  # similar users based on correlation coefficients
    sum_wt = np.sum(similarities) - 1
    product = 1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue
        else:
            product = ratings.iloc[user_loc, indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    # prediction = int(round(wtd_sum / sum_wt))
    prediction = wtd_sum / sum_wt

    # in case of very sparse data sets, using correlation metric for collaborative based approach may give negative ratings
    # which are handled here as below
    if prediction <= 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10

    # print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction))

    return prediction


def predict_rating(ratings_matrix):
    valid_user_id = ratings_matrix.index
    valid_book_id = ratings_matrix.columns
    print("Number of valid users based on collaborative filtering : {}".format(len(valid_user_id)))
    print("Number of valid books based on collaborative filtering : {}".format(len(valid_book_id)))
    out_list = []
    # for i in range(len(valid_user_id)):
    #     for j in range(len(valid_book_id)):
    for i in range(config.cf_users): # taking all valid users based on collaborative filtering
        for j in range(config.cf_books): # taking few valid books based on collaborative filtering
            rating = predict_itembased(valid_user_id[i], valid_book_id[j], ratings_matrix, config.metric,
                                       config.k_neighbour)
            out_list.append({"user_id": valid_user_id[i], "book_id": valid_book_id[j], "rating": rating})

    out_df = pd.DataFrame(out_list)
    out_matrix = out_df.pivot(index='user_id', columns='book_id', values='rating')
    out_matrix.to_csv(config.final_output_item_based_csv, sep=',', encoding="latin-1")

    print("Results written at : {}".format(config.final_output_item_based_csv))


if __name__ == '__main__':
    start = datetime.now()
    ratings_matrix = preprocess()
    predict_rating(ratings_matrix)
    end = datetime.now()
    print("Item based recommendation completed in " + str((end - start)) + " seconds \n")
