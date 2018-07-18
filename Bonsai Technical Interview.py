# import required libraries
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import os.path
import scipy
import json
from scipy import sparse


def calculateSimilarity(popDF):
    '''
    Calculate the column-wise cosine similarity for a sparse matrix.
    Return a new dataframe matrix with similarities.
    
    Inputs:
    
    popDF (DataFrame): User-Item ratings dataframe
    
    Outputs:
    
    sim (DataFrame): Item-Item Similarity dataframe
    '''
    sparseData = sparse.csr_matrix(popDF)
    similarities = cosine_similarity(sparseData.transpose())
    sim = pd.DataFrame(data=similarities, index=popDF.columns, columns= popDF.columns)
    return sim


def predictByItemSimilarity(trainSet, numUsers, numItems, similarity):
    
    '''
    Uses the similarity matrix to predict rating scores for every user for each item.
    
    Inputs:
    
    trainSet (Numpy Array): Item-User matrix
    numUsers (int): Number of users
    numItems (int): Number of items
    similarity (Numpy Array): Item-Item similarity matrix
    
    Ouput:
    
    predictionMatrix (Numpy Array): User-Item matrix with predicted ratings for each user and product.
    
    '''
    # Initialize the predicted rating matrix with zeros
    predictionMatrix = np.zeros((numItems, numUsers))
    
    for (item,user), rating in np.ndenumerate(trainSet):
        # Predict rating for every item that wasn't ranked by the user (rating == 0)
        if rating == 0:
            # Extract the items that were bought by users
            itemVector = trainSet[:,user]
            
            itemRatings = itemVector[itemVector.nonzero()]
            
            # Get the similarity score for each of the items that were bought by users
           
            itemsSim = similarity[item,:][itemVector.nonzero()]
            # If there no users bought any items, use item's average
            if len(itemsSim) == 0:
                itemVector = trainSet[item, :]
                ratedItems = itemVector[itemVector.nonzero()]
                
                # If the items werent rated use 0, otherwise use average
                if len(ratedItems) == 0:
                    predictionMatrix[item,user] = 0
                else:
                    predictionMatrix[item,user] = ratedItems.mean()
            else:
                # predict score based on item-item similarity
                if(itemsSim.sum() == 0):
                    predictionMatrix[item,user] = 0
                else:
                    predictionMatrix[item,user] = (itemRatings*itemsSim).sum() / itemsSim.sum()
        
        # report progress every 100 items
        if (item % 100 == 0 and user == 1):
            print ("calculated %d items" % (item,))
    

    return predictionMatrix

def main():
    with open('training_mixpanel.txt') as f:
        data = json.load(f)

    #Initialize customer dataframe
    custDF = json_normalize(data)
    custDF = custDF[custDF['properties.quantity'] > 0]
    custDF = custDF.drop_duplicates()

    #Create column item popularity rating.
    custDF['popRating'] = 1 + np.log(np.array(custDF['properties.quantity']) * np.array(custDF['properties.unit_price']))

    #Get average rating for each item per customer.
    avgCustRating = custDF.groupby(['properties.customer_id', 'properties.product_id'])['popRating'].mean().reset_index()
    avgCustRating = avgCustRating.replace([-np.inf], 0)

    index = [tuple(item) for item in avgCustRating.values]

    popDF = pd.DataFrame(0,columns=custDF['properties.product_id'].unique(), index=custDF['properties.customer_id'].unique())
    popDF = popDF.astype(float)

    #Populate user-item matrix with the user-item rating.
    for item in index:
        popDF.at[item[0], item[1]] = float(item[2])

    #Calculate cosine item similarity coefficients
    itemSimilarity = calculateSimilarity(popDF) #pairwise_distances(trainMatrix.T, metric='cosine')

    numItems = popDF.shape[1]
    numCustomers = popDF.shape[0]

    #Calculate user-item prediction ratings
    predictionMatrix = predictByItemSimilarity(popDF.T.as_matrix(), numCustomers, numItems, itemSimilarity.as_matrix())
    predictionDF = pd.DataFrame(predictionMatrix.T, columns=popDF.columns, index=popDF.index)
    predictionDF.to_csv('output.csv')


if __name__ == '__main__':
    main()




















