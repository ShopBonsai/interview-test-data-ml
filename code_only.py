from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy import interp
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def fix_users_dataframe(users_df):
    users_df['user'] = users_df['user'].apply(lambda x: str(x).split('.')[0])
    users_df['location'].fillna('', inplace=True)
    users_df['location'] = users_df['location'].apply(lambda x: x.replace(',', ''))
    users_df['age'] = users_df['age'].astype(float)
    users_df['age'].fillna(0.0, inplace=True)
    users_df['age_defined'] = ~(users_df['age'] == 0.0)
    users_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    return users_df


def fix_books_dataframe(books_df):
    books_df['author'].fillna('', inplace=True)
    books_df.drop(['urlId', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True) #We are not using the urlid field
    return books_df


def fix_user_events_dataframe(user_events_df):
    user_events_df.rename(index=str, columns={'Unnamed: 0': 'order'}, inplace=True)
    user_events_df['user'] = user_events['user'].astype(str)
    return user_events_df


def create_book_info_dictionary(book_dataframe):
    book_dict = {}
    for _, row in book_dataframe.iterrows():
        book_dict[row['bookISBN']] = {'author': row['author'],
                                      'name': row['bookName'],
                                      'year': row['yearOfPublication'],
                                      'publisher': row['publisher']
                                      }
    return book_dict


def create_user_info_dictionary(user_dataframe):
    user_dict = {}
    for _, row in user_dataframe.iterrows():
        user_dict[row['user']] = {'age': row['age'],
                                  'location': row['location'],
                                  'age_defined': int(row['age_defined'])}
    return user_dict


def append_number_in_dictionary(my_dict, my_key, value, avoid_zeros=True):
    if value == 0 and avoid_zeros:
        return

    if my_key in my_dict:
        my_dict[my_key].append(float(value))
    else:
        my_dict[my_key] = [float(value)]


def append_string_in_dictionary(my_dict, my_key, value):
    if my_key in my_dict:
        my_dict[my_key] += ' ' + str(value)
    else:
        my_dict[my_key] = str(value)


def get_valid_books(user_events_df, books_df, valid_users):
    set_usable_book_ids = set(user_events_df[(user_events_df['impression'] == 'checkout') &
                                         (user_events_df['user'].isin(valid_users))]['bookId'])
    set_known_book_ids = set(books_df['bookISBN'])
    set_books_valid_users = list(set_usable_book_ids.intersection(set_known_book_ids))
    return set_books_valid_users


def get_num_purchases_by_book(user_events_df, valid_users, valid_books):
    print('Verifying how many times each book was bought')
    num_sales = {}
    filtered_events_df = user_events_df[(user_events_df['impression'] == 'checkout')
                                      & (user_events_df['user'].isin(valid_users))]
    for book in tqdm(valid_books):
        relevant_events = filtered_events_df[filtered_events_df['bookId'] == book]
        current_num_sales = relevant_events.shape[0]
        num_sales[book] = current_num_sales
    return num_sales


def build_prototype_dict(existent_impressions):
    prototype_dict = {}

    for impr in existent_impressions:
        impr_id = impr + '_id'
        impr_name = impr + '_name'
        impr_author = impr + '_author'
        impr_year = impr + '_year'
        impr_publisher = impr + '_publisher'

        prototype_dict[impr_id] = ''
        prototype_dict[impr_name] = ''
        prototype_dict[impr_author] = ''
        prototype_dict[impr_year] = []
        prototype_dict[impr_publisher] = ''

    prototype_dict.update({'age': 0.0, 'location': '', 'age_defined': 0})
    return prototype_dict


def make_mean_of_year_features(feature_dict):
    target_keys = [k for k in feature_dict.keys() if '_year' in k]
    for key in target_keys:
        if len(feature_dict[key]) > 0:
            feature_dict[key] = np.mean(feature_dict[key])
        else:
            feature_dict[key] = 0.0


def build_user_purchase_info_dict(user,
                                   impressions_available,
                                   user_info_dict,
                                   book_info_dict,
                                   number_of_previous_events=10):
    """
    The featureset is built as follows:

    We first divide the books that the user interacted with by impression (like, dislike, add to cart).
    For each group, we concatenate the all the book's ids, all the book's names, 
    all the book's authors and all the book's publishers. Each one of these
    strings will be futher vectorized using the bag of words strategy. We also compute the mean
    of the publication years of the books.

    The user's age and location is also used as feature. If the user is not known (is not present on the user's
    dataframe), the age is set to zero, the location is set to the empty string and a flag called age_defined will be
    set to zero (it is one on the known users)
    """

    if type(user) != str:
        user = str(user)

    try:
        user_info = user_info_dict[user]
        user_age = user_info['age']
        location = user_info['location']
        age_defined = user_info['age_defined']
    except:
        user_age = 0.0
        location = ''
        age_defined = 0

    relevant_events = user_events[user_events['user'] == user]
    relevant_events = relevant_events.sort_values(by='order')
    relevant_purcharces = relevant_events[relevant_events['impression'] == 'checkout']
    features = []

    for _, current_purcharce in relevant_purcharces.iterrows():
        prototype = build_prototype_dict(impressions_available)

        current_order = current_purcharce['order']
        current_bid = current_purcharce['bookId']
        happened_before = relevant_events[relevant_events['order'] < current_order].tail(number_of_previous_events)

        for _, row in happened_before.iterrows():
            impr = row['impression']
            current_book = row['bookId']

            impr_id = impr + '_id'
            impr_name = impr + '_name'
            impr_author = impr + '_author'
            impr_year = impr + '_year'
            impr_publisher = impr + '_publisher'

            book_name = ''
            book_author = ''
            book_year = 0.0
            book_publisher = ''

            if current_book in book_info_dict:
                book_info = book_info_dict[current_book]
                book_name = book_info['name']
                book_author = book_info['author']
                book_year = book_info['year']
                book_publisher = book_info['publisher']

            append_string_in_dictionary(prototype, impr_id, current_book)
            append_string_in_dictionary(prototype, impr_name, book_name)
            append_string_in_dictionary(prototype, impr_author, book_author)
            append_number_in_dictionary(prototype, impr_year, book_year)
            append_string_in_dictionary(prototype, impr_publisher, book_publisher)

        make_mean_of_year_features(prototype)
        prototype.update({'age': user_age,
                          'location': location,
                          'age_defined': age_defined})
        features.append({'raw_features': prototype,
                         'user_id': user,
                         'book_bought': current_bid})
    return features


def create_purchase_info_for_user_set(user_set):
    purchase_info = {}
    num_feats = 0

    print('Creating feature dicts')
    for user in tqdm(user_set):
        new_features = build_user_purchase_info_dict(user,
                                                      set_of_impressions,
                                                      user_dict,
                                                      book_dict)
        for new_feat in new_features:
            if new_feat['book_bought'] not in set_valid_books:
                continue

            num_feats += 1

            if new_feat['book_bought'] in purchase_info:
                purchase_info[new_feat['book_bought']].append(new_feat)
            else:
                purchase_info[new_feat['book_bought']] = [new_feat]

    print('Num feature dicts', num_feats)

    return purchase_info


def separate_train_and_test_purchase_info(all_purchase_info):
    purchase_dicts_train = []
    purchase_dicts_test = []
    print('Separating into train and test sets')
    for book_key in tqdm(all_purchase_info.keys()):
        total = len(all_purchase_info[book_key])
        num_train = max(int(0.8 * total), 1)
        purchase_dicts_train += all_purchase_info[book_key][:num_train]
        purchase_dicts_test += all_purchase_info[book_key][num_train:]

    print('Num feats train', len(purchase_dicts_train))
    print('Num feats test', len(purchase_dicts_test))

    return purchase_dicts_train, purchase_dicts_test


def create_vectorizers_and_statistics(purchase_informations):
    name_vectorizer = CountVectorizer()
    author_vectorizer = CountVectorizer()
    publisher_vectorizer = CountVectorizer()
    id_vectorizer = CountVectorizer()
    location_vectorizer = CountVectorizer()

    ages = [p['raw_features']['age'] for p in purchase_informations]
    locations = [p['raw_features']['location'] for p in purchase_informations]

    names_texts = []
    authors_texts = []
    publishers_texts = []
    ids_texts = []
    years = []

    feat_names = list(purchase_informations[0]['raw_features'].keys())
    for fn in feat_names:
        for pi in purchase_informations:
            pf = pi['raw_features']
            if '_name' in fn:
                names_texts.append(pf[fn])
            if '_author' in fn:
                authors_texts.append(pf[fn])
            if '_publisher' in fn:
                publishers_texts.append(pf[fn])
            if '_id' in fn:
                ids_texts.append(pf[fn])
            if '_year' in fn:
                years.append(pf[fn])

    statistics_age = {'max': np.max(ages), 'min': np.min(ages)}
    statistics_years = {'max': np.max(years), 'min': np.min(years)}

    name_vectorizer.fit(names_texts)
    author_vectorizer.fit(authors_texts)
    publisher_vectorizer.fit(publishers_texts)
    id_vectorizer.fit(ids_texts)
    location_vectorizer.fit(locations)

    return {
        'name_vectorizer': name_vectorizer,
        'author_vectorizer': author_vectorizer,
        'publisher_vectorizer': publisher_vectorizer,
        'id_vectorizer': id_vectorizer,
        'location_vectorizer': location_vectorizer,
        'statistics_age': statistics_age,
        'statistics_years': statistics_years
    }


def feature_dict_to_feature_vector(feature_dict,
                                   vectorizers_and_statistics):
    id_vectorizer = vectorizers_and_statistics['id_vectorizer']
    name_vectorizer = vectorizers_and_statistics['name_vectorizer']
    author_vectorizer = vectorizers_and_statistics['author_vectorizer']
    publisher_vectorizer = vectorizers_and_statistics['publisher_vectorizer']
    location_vectorizer = vectorizers_and_statistics['location_vectorizer']
    statistics_age = vectorizers_and_statistics['statistics_age']
    statistics_years = vectorizers_and_statistics['statistics_years']
    feature_vectors = []
    normalized_age = (feature_dict['age'] - statistics_age['min']) / (statistics_age['max'] - statistics_age['min'])
    feature_vectors.append(normalized_age)
    feature_vectors.append(feature_dict['age_defined'])

    for impr in set_of_impressions:
        relevant_feat_names = [k for k in feature_dict.keys() if impr in k]
        for fn in relevant_feat_names:
            if '_id' in fn:
                feature_vectors.append(id_vectorizer.transform([feature_dict[fn]])[0])
            if '_name' in fn:
                feature_vectors.append(name_vectorizer.transform([feature_dict[fn]])[0])
            if '_author' in fn:
                feature_vectors.append(author_vectorizer.transform([feature_dict[fn]])[0])
            if '_publisher' in fn:
                feature_vectors.append(publisher_vectorizer.transform([feature_dict[fn]])[0])
            if '_year' in fn:
                normalized_val = (feature_dict[fn] - statistics_years['min']) / \
                                 (statistics_years['max'] - statistics_years['min'])
                feature_vectors.append([normalized_val])

    feature_vectors.append(location_vectorizer.transform([feature_dict['location']])[0])

    return hstack(feature_vectors)


def build_numeric_featureset_and_targets(purchase_infos, vectorizers_and_statistics, target_name_list):
    X = []
    Y = []
    USERS = []

    print('Building the feature vectors')
    for pi in tqdm(purchase_infos):
        x = feature_dict_to_feature_vector(pi['raw_features'], vectorizers_and_statistics)
        X.append(x)
        Y.append(pi['book_bought'])
        USERS.append(pi['user_id'])

    X = vstack(X)
    Y = [target_name_list.index(y) for y in Y]

    return X, Y, USERS


def generate_result_dataframe(model, X_test, Y_test, Users_test, target_name_list):
    column_names = ['book_' + tg for tg in target_name_list]
    y_score = model.predict_proba(X_test)
    result_df = pd.DataFrame(y_score, columns=column_names)
    result_df.insert(0, 'user_id', Users_test)
    return result_df


def save_results(result_df):
    matrix_with_no_user = result_df.drop(['user_id'], axis=1)
    matrix_with_no_user.to_csv('result_matrix.csv', sep=';', index=False, header=False)


# Based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def compute_mean_auroc(result_dataframe, Y_test, plot_mean_roc=True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    result_dataframe = result_dataframe.drop(['user_id'], axis=1)
    num_classes = result_dataframe.shape[1]
    result_matrix = result_dataframe.values

    for c in tqdm(range(num_classes)):
        current_Y_test = [int(y == c) for y in Y_test]
        current_Y_score = result_matrix[:, c]
        fpr[c], tpr[c], _ = roc_curve(current_Y_test, current_Y_score)
        roc_auc[c] = auc(fpr[c], tpr[c])

    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[c], tpr[c])

    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if plot_mean_roc:
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"],
                 color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    return roc_auc['macro']

books = pd.read_csv(os.path.join("book-recommend-data", "Books.csv"), sep=',', header=0, encoding="latin1")
users = pd.read_csv(os.path.join("book-recommend-data", "Users.csv"), sep=',', header=0, encoding="latin1")
user_events = pd.read_csv(os.path.join("book-recommend-data", "UserEvents.csv"), sep=',', header=0, encoding="latin1")

books = fix_books_dataframe(books)
users = fix_users_dataframe(users)
user_events = fix_user_events_dataframe(user_events)

set_all_users = set(user_events['user'])
set_books = get_valid_books(user_events, books, set_all_users)
print('Number of different known books that were bought by a known user', len(set_books))

#Verifying book relevance
num_sales = get_num_purchases_by_book(user_events, set_all_users, set_books)

#For now, let's limitate our analysis to the books who were bought at least 10 times.
#We will also not limitate the users to the ones present at the User's dataframe
minimum_number_of_sales = 10
set_valid_books = list([k for k in num_sales.keys() if num_sales[k] >= minimum_number_of_sales])
print('Number of valid books', len(set_valid_books))

#Let's check which users have bought known books
set_valid_users = list(user_events[user_events['bookId'].isin(set_valid_books)
                                  & (user_events['impression'] == 'checkout')]['user'].unique())
print('Number of usable users', len(set_valid_users))

set_of_impressions = list(user_events['impression'].unique())
book_dict = create_book_info_dictionary(books)
user_dict = create_user_info_dictionary(users)

all_purchase_info = create_purchase_info_for_user_set(set_valid_users)
purchase_dicts_train, purchase_dicts_test = separate_train_and_test_purchase_info(all_purchase_info)

vectorizers_and_statistics = create_vectorizers_and_statistics(purchase_dicts_train)

print('TRAIN SET')
X_train, Y_train, Users_train = build_numeric_featureset_and_targets(purchase_dicts_train,
                                                                     vectorizers_and_statistics,
                                                                     set_valid_books)

print('TEST SET')
X_test, Y_test, Users_test = build_numeric_featureset_and_targets(purchase_dicts_test,
                                                                  vectorizers_and_statistics,
                                                                  set_valid_books)

print('TRAINING CLASSIFIER')
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

print('CALCULATING AND SAVING THE OUTPUT MATRIX')
rm = generate_result_dataframe(classifier, X_test, Y_test, Users_test, set_valid_books)
save_results(rm)

print('CALCULATING THE AUROC')
auroc = compute_mean_auroc(rm, Y_test)
print('Average AOC', auroc)
