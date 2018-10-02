import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np


class Utils():
    def transform_user_impression_to_rating(self, row):
        mapping = {"dislike": 1, "view": 2, "interact": 3, "like": 4, "add to cart": 5, "checkout": 6}
        return mapping[row["impression"]]

    def clean_book_dataframe(self, books):
        # dropping urlId which is an useful column
        books.drop(["urlId", "Unnamed: 0"], axis=1, inplace=True)
        # print(books.loc[books.yearOfPublication == 'DK Publishing Inc',:])
        # bookISBN '0789466953'
        books.loc[books.bookISBN == '0789466953', 'yearOfPublication'] = 2000
        books.loc[books.bookISBN == '0789466953', 'author'] = "James Buckley"
        books.loc[books.bookISBN == '0789466953', 'publisher'] = "DK Publishing Inc"
        books.loc[books.bookISBN == '0789466953', 'bookName'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
        # bookISBN '078946697X'
        books.loc[books.bookISBN == '078946697X', 'yearOfPublication'] = 2000
        books.loc[books.bookISBN == '078946697X', 'author'] = "Michael Teitelbaum"
        books.loc[books.bookISBN == '078946697X', 'publisher'] = "DK Publishing Inc"
        books.loc[books.bookISBN == '078946697X', 'bookName'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"


        books.yearOfPublication = pd.to_numeric(books.yearOfPublication, errors='coerce')

        books.loc[(books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
        # replacing NaNs with mean value of yearOfPublication
        books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)

        # resetting the dtype as int32
        books.yearOfPublication = books.yearOfPublication.astype(np.int32)
        books.loc[(books.bookISBN == '1931696993'), 'publisher'] = 'other'

        return books

    def clean_user_dataframe(self, users):
        # print(sorted(users.age.unique()))
        # ASSUMPTION : values below 5 and above 90 do not make much sense for our book rating case, hence replacing these by NaNs
        users.loc[(users.age > 90) | (users.age < 5), 'age'] = np.nan

        # replacing NaNs with mean
        users.age = users.age.fillna(users.age.mean())
        users.user = users.user.fillna(0)
        # setting the data type as int
        users.age = users.age.astype(np.int32)
        users.user = users.user.astype(np.int32)
        users.drop(["Unnamed: 0"], axis=1, inplace=True)

        return users

    def clean_events_dataframe(self, ratings):

        # MAPPING THE IMPRESSION TO RATING
        ratings['bookRating'] = ratings.apply(lambda row: self.transform_user_impression_to_rating(row), axis=1)
        ratings.drop(["impression", "Unnamed: 0"], axis=1, inplace=True)

        return ratings



