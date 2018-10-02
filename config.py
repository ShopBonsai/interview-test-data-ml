import os
class Config():

    book_csv = "C:/Documents/Canada/ShopBonsai/book_recommendation/Books.csv"
    user_csv = "C:/Documents/Canada/ShopBonsai/book_recommendation/Users.csv"
    user_event_csv = "C:/Documents/Canada/ShopBonsai/book_recommendation/UserEvents.csv"

    metric = "cosine"
    k_neighbour = 10

    final_output_user_based_csv = "C:/Documents/Canada/ShopBonsai/book_recommendation/result/result_user_based.csv"
    final_output_item_based_csv = "C:/Documents/Canada/ShopBonsai/book_recommendation/result/result_item_based.csv"

    cf_constant = 100 # considering users who have rated at least 100 books and books which have at least 100 ratings

    # due to limited capability of my system, the final ratings will for 127 users and 150 books only
    # MXN = 127 X 150
    cf_users = 127
    cf_books = 150
