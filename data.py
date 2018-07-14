import json
import pandas as pd
import math
import itertools


def extract_data_from_file():
    """
    Extracts user, item data. Infers User preference (score) using  a function of cost and number of times purchased
    :return: dataframe
    """
    file_name = "training_mixpanel.txt"
    rows = []
    scores_skipped = 0
    try:
        try:
            with open(file_name) as f:
                data = json.load(f)
        finally:
            f.close()

        for i in data:
            row = {}
            row['event'] = i['event']
            for j in i['properties']:
                row[j] = i['properties'][j]
            try:
                if row['quantity'] > 0:
                    row['score'] = math.log(1 + (row['quantity'] * row['unit_price']))
                    rows.append(row)
            except Exception as e:
                scores_skipped += 1
                print('Negative Feedback ... omitting transacton.')

    except Exception as e:
        print(e)
    df = (pd.DataFrame(rows))
    return df


def write_training_datasets(df):
    """
    Writes the datasets that are used for training
    #1 users.csv Dataset containing user features, user location is used as feature
    #2 data.csv Recommender dataset containing user-item-score
    :param df:
    :return: dataframe
    """
    df_users = df.drop_duplicates('customer_id')
    df_users = df_users[['customer_id', 'country']]
    df_users.to_csv("users.csv", index=False)
    df = df[['customer_id', 'product_id', 'score']]
    df.to_csv("data.csv", index=False)
    return df


def generate_test_dataset_to_predict(df):
    """
    Generates all user-item combinations for prediction and writes it as a csv file
    :param df:
    :return:
    """
    tuples = []
    cnt = 0
    for i in itertools.product(set(df['customer_id'].tolist()), set(df['product_id'].tolist())):
        tuples.append(i)
        cnt += 1
        if cnt % 100000 == 0:
            print('Processed entries: %d'%cnt)
    pd.DataFrame(tuples, columns=['customer_id', 'product_id']).to_csv("testSet.csv", index=False)


if __name__ == "__main__":
    df = extract_data_from_file()
    df = write_training_datasets(df)
    generate_test_dataset_to_predict(df)

    # once data sets are constructed, hybrid recommender model is created using Matchbox Algorithm
    # Algorithm is found here: https://gallery.cortanaintelligence.com/Experiment/test-reco
    # Data were split as 75-25 for algo training, achieved RMSE 0.78
    # results are written in predictions.csv
    # Rscript predictions.R is used to transform data as per the test requirement
