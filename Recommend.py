import json
import numpy as np
import pandas as pd
import math

pd.options.mode.chained_assignment = None

'''
This simple model uses Item-based collaborate filter algorithm:
    1. The recommendation is based on the item similarity, the result is the multiplication 
    of user-item frequency matrix and item similarity matrix. I choose top ranking 20 items 
    from 3 most similar users for each user according to the preference scores to reduce the
    workload for recommendation.

    2. Due to the limited time, 
    I only use the frequency of customer making purchase of the item as the initial score.
    Future work:
    The item similarity based on Description(using TF-IDF) can be combined with 
    the cosine similarity of two items to increase the accuracy of recommendation.
    Also, the average number of items bought for each user can also be counted as the weight of 
    a user's preference.
    To reduce the running time, it is better to use map reduce principle.
'''


def read_data(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
        propertiesList = [e['properties'] for e in data]
        PropertiesDF = pd.DataFrame.from_dict(propertiesList, orient='columns')
        # print(PropertiesDF,PropertiesDF.shape)
    return PropertiesDF


def createFrequencyScore(PropertiesDF):
    userItemDF = PropertiesDF[['customer_id', 'product_id']]
    userItemDF['frequency'] = userItemDF.groupby('product_id')['product_id'].transform('count')
    train = dict()
    for row in userItemDF.iterrows():
        index, data = row
        user, item, fre = data[0], data[1], data[2]
        train.setdefault(user, {})
        train[user][item] = float(fre)

    # get lists of unique items and users
    userList = list(set(userItemDF["customer_id"].tolist()))
    itemList = list(set(userItemDF["product_id"].tolist()))

    return train, userList, itemList


def ItemSimilarity(train_Dict):
    # Calculate item-item CF matrix
    C = dict()
    # number of users for an unique item
    N = dict()
    for user, items in train_Dict.items():
        for i in items.keys():
            N.setdefault(i, 0)
            N[i] += 1
            C.setdefault(i, {})
            for j in items.keys():
                if i == j: continue
                C[i].setdefault(j, 0)
                C[i][j] += 1
    # similarity matrix
    W = dict()
    for i, related_items in C.items():
        W.setdefault(i, {})
        for j, cij in related_items.items():
            W[i][j] = cij / (math.sqrt(N[i] * N[j]))
    return W


def Recommend(user, train, K=3, N=20):
    rank = dict()
    W = ItemSimilarity(train)
    action_item = train[user]  # item related to users and their scores
    for item, score in action_item.items():
        for j, wj in sorted(W[item].items(), key=lambda x: x[1], reverse=True)[0:K]:
            if j in action_item.keys():
                continue
            rank.setdefault(j, 0)
            rank[j] += score * wj
    return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


def getRecommendMatrix(userList, itemList, trainDict):
    # Get count of users and items
    userCount = len(userList)
    itemCount = len(itemList)

    # MxN matrix represents the preference score
    result = np.zeros([userCount, itemCount])
    for user in range(userCount):
        recommedDic = Recommend(userList[user], trainDict)
        print(recommedDic)
        for item in range(itemCount):
            for key, value in recommedDic.items():
                if key == itemList[item]:
                    result[user][item] = value
    np.savetxt("result.csv", result, delimiter=",")


if __name__ == '__main__':
    File = 'training_mixpanel.txt'
    PropertiesDF = read_data(File)
    train, userList, itemList = createFrequencyScore(PropertiesDF)
    getRecommendMatrix(userList, itemList, train)
