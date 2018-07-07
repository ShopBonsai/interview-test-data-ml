import json
import numpy as np
import pandas as pd
import surprise as s
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import SVD



### 1. DATA

with open('training_mixpanel.txt') as rawdata:
    dicts = json.load(rawdata)

properties_list = []
for i in range(len(dicts)):
    properties_list.append(dicts[i].get('properties'))
properties = pd.DataFrame(properties_list)



### 2. PREPROCESS

# drop duplicate rows
properties = properties.drop_duplicates()

properties = properties.drop(['country', 'description', 'invoice_date'], axis=1)

def rate_of_return(df=properties):

    ''' The ratio of purchases to returns of each product. '''
    
    # first, add a column indicating the return count of an item on a single invoice
    df.loc[df['quantity'] >= 0, 'product_return_count'] = 0
    df.loc[df['quantity'] < 0, 'product_return_count'] = -1*df['quantity']
    # next, add a column indicating the total number of times an item was returned
    num_product_returned = df.groupby('product_id')['product_return_count'].sum()
    df = df.join(num_product_returned.to_frame(name='num_product_returned'), on='product_id')
    
    # disregard information on returned products (just for the below aggregate)
    products_purchased = df.loc[df['quantity'] >= 0]
    # then, add a column containing the total number of times an item was purchased
    num_product_purchased = products_purchased.groupby(['product_id'])['quantity'].sum()
    df = df.join(num_product_purchased.to_frame(name='num_product_purchased'), on='product_id')

    # replace null values with 0, indicating this product has no purchase info (purchased 0 times)
    df['num_product_purchased'] = df['num_product_purchased'].fillna(0)
    # define product return rate; note that there will be division by 0
    df['product_return_rate'] = df['num_product_returned'] / df['num_product_purchased']
    # and replace inf (from previous division by 0) with null
    df['product_return_rate'] = df['product_return_rate'].replace([np.inf,-np.inf], np.nan)

    # replace null values with the maximal return rate
    max_return_rate = df['product_return_rate'].max()
    df['product_return_rate'] = df['product_return_rate'].fillna(max_return_rate)
    
    # finally, drop the new columns that aren't needed (num_product_purchased will be used later)
    df = df.drop(['product_return_count','num_product_returned'], axis=1)

    return df

properties = rate_of_return()



### 3. POPULARITY

def product_popularity(df=properties):
    
    ''' The popularity of a product is a function of the purchase count and return rate. '''

    # popularity = (number of purchases) x (1 - return rate)
    df['popularity'] = df['product_return_rate'].apply(lambda x: 1-x) * df['num_product_purchased']
    # rescale popularity to get rating
    df['popularity_rating'] = (df['popularity']-df['popularity'].min()) / (df['popularity'].max()-df['popularity'].min())
    
    # introduce column for the number of unique products purchased by a customer
    num_unique_products = df.groupby('customer_id')['product_id'].nunique()
    df = df.join(num_unique_products.to_frame(name='customer_unique_products'), on='customer_id')
        
    df = df.drop(['popularity', 'num_product_purchased'], axis=1)
        
    return df

properties = product_popularity()


def clip_and_scale_uniquecount(df=properties):

    ''' Clips and scales the number of unique product purchases of each user. '''
    
    # clip values outside boundary to boundary values 
    df['customer_unique_products'].clip(lower=1, upper=100, inplace=True)
    # log scale
    df['customer_unique_products_logscale'] = df['customer_unique_products'].apply(lambda x: np.log10(x))
    
    # subtract values by their maximum; global variable to be used later as well
    global logscale_max
    logscale_max = df['customer_unique_products_logscale'].max()
    df['popularity_weight'] = df['customer_unique_products_logscale'].apply(lambda x: logscale_max-x)
    
    df = df.drop(['customer_unique_products','customer_unique_products_logscale'], axis=1)
    
    return df

properties = clip_and_scale_uniquecount()


# table of users and their associated rating weights
rating_weights = properties.groupby('customer_id')['popularity_weight'].unique().reset_index()
rating_weights['popularity_weight'] = rating_weights['popularity_weight'].astype(float)

# table of products and their associated popularity ratings
popularity_ratings = properties.groupby('product_id')['popularity_rating'].unique().reset_index()
popularity_ratings['popularity_rating'] = popularity_ratings['popularity_rating'].astype(float)

# MxN matrix of user-product ratings
weighted_popularity_matrix = np.outer(np.array(rating_weights['popularity_weight']),
                                      np.array(popularity_ratings['popularity_rating']))

# sanity check for matrix shape
print('customers in dataset:', properties['customer_id'].nunique())
print('products in dataset:', properties['product_id'].nunique())
print('shape of popularity-ratings matrix:', weighted_popularity_matrix.shape)
print('')



### 4. COLLABORATIVE-FILTERING

def interaction_threshold(df=properties, threshold=3):

    ''' Creates a dataframe with users that have bought more than "threshold" products. '''

    num_user_interactions = df.groupby(['customer_id','product_id']).size().groupby('customer_id').size()
    print('users: %d' % len(num_user_interactions))
    above_threshold_user_interactions = num_user_interactions[num_user_interactions >= threshold].reset_index()[['customer_id']]
    print('users that have purchased at least', threshold, 'items: %d' % len(above_threshold_user_interactions))
    
    # list of users that are below threshold; global variable will be used later
    global below_threshold_user_interactions
    below_threshold_user_interactions = num_user_interactions[num_user_interactions < threshold].reset_index()[['customer_id']]
    print('users that have purchased fewer than', threshold, 'items: %d' % len(below_threshold_user_interactions))

    # thresholded dataframe obtained from properties dataframe
    df_thresholded = df.merge(above_threshold_user_interactions, 
                              how='right', 
                              left_on='customer_id', 
                              right_on='customer_id')
    print('')
    print('products: %d' % df['product_id'].nunique())
    print('products after threshold: %d' % df_thresholded['product_id'].nunique())
    print('interactions: %d' % len(df))
    print('interactions after threshold: %d' % len(df_thresholded))    
    
    return df_thresholded

properties_thresh = interaction_threshold()


# create a dataframe of unique user-item pairs, along with quantity x price
properties_thresh['total_price'] = properties_thresh['quantity'] * properties_thresh['unit_price']
properties_thresh = properties_thresh.drop(['quantity','unit_price'], axis=1)
customer_product_interactions = properties_thresh.groupby(['customer_id', 'product_id'])['total_price'].sum().reset_index()
customer_product_interactions.rename(columns={'total_price':'total_user_product_price'}, inplace=True)


def clip_and_scale_interactions(df=customer_product_interactions, num_std=0.1):
    
    ''' Clips and scales the rating for each user-product pair. '''
    
    col = 'total_user_product_price'
    average = df[col].mean()
    threshold = df[col].std() * num_std
    # clip values outside boundary to boundary values
    df[col].clip(lower=-threshold, upper=average+threshold, inplace=True)
    # rescale between [0,1]
    df['scaled_user_product_price'] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
    df = df.drop(col, axis=1)
    
    return df

customer_product_interactions = clip_and_scale_interactions()


# loading the above data in a manner required by the surprise library
data = s.Dataset.load_from_df(customer_product_interactions[['customer_id', 'product_id', 'scaled_user_product_price']],
                              rating_scale=(0,1))

# simple fit using the SVD algorithm provided with surprise
trainset, testset = train_test_split(data, test_size=.2)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
print('')
s.accuracy.rmse(predictions)


# put all customer-product pairs in a dataframe
customers_df_thresh = properties_thresh.customer_id.to_frame(name='customer_id')
customers_df_thresh['key'] = 0
customers_df_thresh = customers_df_thresh.drop_duplicates()
products_df_thresh = properties_thresh.product_id.to_frame(name='product_id')
products_df_thresh['key'] = 0
products_df_thresh = products_df_thresh.drop_duplicates()
full_customer_product_thresh = pd.merge(customers_df_thresh, products_df_thresh, on='key').drop('key', axis=1)
full_customer_product_thresh['collabfilter_ratings'] = 0

# load above dataframe into surprise and convert to numpy array
data2 = s.Dataset.load_from_df(full_customer_product_thresh[['customer_id', 'product_id', 'collabfilter_ratings']],
                              rating_scale=(0,1))
data2 = np.array(full_customer_product_thresh)

all_collabfilter_predictions = algo.test(data2)
all_collabfilter_predictions = np.array(all_collabfilter_predictions)


# extract the first, second, and fourth slices from this array:
# (customer, product, and predicted rating)
customer_product_ratings_collabfilter = all_collabfilter_predictions[:,[0,1,3]]

# convert to dataframe
customer_product_ratings_collabfilter = pd.DataFrame(data=customer_product_ratings_collabfilter,
                                                     columns=['customer_id','product_id','collabfilter_ratings'])

# change the datatype of "customer_id" and "collabfilter_ratings" to int and float, respectively
customer_product_ratings_collabfilter['customer_id'] = customer_product_ratings_collabfilter['customer_id'].astype(int)
customer_product_ratings_collabfilter['collabfilter_ratings'] = customer_product_ratings_collabfilter['collabfilter_ratings'].astype(float)


# rating weights for collaborative filtering
collabfilter_weights = rating_weights
collabfilter_weights['cf_weight'] = collabfilter_weights['popularity_weight'].apply(lambda x: logscale_max - x)
collabfilter_weights = collabfilter_weights.drop('popularity_weight', axis=1)

# merge ratings and weights data
weighted_collabfilter_df = pd.merge(customer_product_ratings_collabfilter, 
                                     collabfilter_weights, 
                                     on='customer_id', 
                                     how='left')

# scale ratings by weights
weighted_collabfilter_df['weighted_cf_ratings'] = weighted_collabfilter_df['collabfilter_ratings'] * weighted_collabfilter_df['cf_weight']
weighted_collabfilter_df = weighted_collabfilter_df.drop(['collabfilter_ratings','cf_weight'], axis=1)


# dataframe containing only those rows where users are below threshold of unique product interactions
properties_below_thresh = pd.merge(properties, 
                                   below_threshold_user_interactions, 
                                   on='customer_id')
customers_df_below_thresh = properties_below_thresh.customer_id.to_frame(name='customer_id')
customers_df_below_thresh['key'] = 0
customers_df_below_thresh = customers_df_below_thresh.drop_duplicates()

# put all customer-product pairs in a dataframe
# (below threshold customers only, but using all products)
full_customer_product_below_thresh = pd.merge(customers_df_below_thresh, 
                                            products_df_thresh, # all products
                                            on='key').drop('key', axis=1)
# naming the column the same as the previous dataframe so that we can easily append it there
full_customer_product_below_thresh['weighted_cf_ratings'] = 0


# append below threshold user-product pairs to above threshold
weighted_collabfilter_df = weighted_collabfilter_df.append(full_customer_product_below_thresh, sort=False)
# sort first by customer_id then by product_id to ensure correct ordering of ratings
weighted_collabfilter_df = weighted_collabfilter_df.sort_values(by=['customer_id','product_id'])


# unique products and customers in original dataset
M = properties.customer_id.nunique()
N = properties.product_id.nunique()
# weighted matrix of ratings determined by collaborative filtering
weighted_collabfilter_matrix = np.array(weighted_collabfilter_df.weighted_cf_ratings)
weighted_collabfilter_matrix = weighted_collabfilter_matrix.reshape(M,N)
print('shape of collab-filter matrix:', weighted_collabfilter_matrix.shape)

# final ratings are a weighted average
final_ratings = 0.5*np.add(weighted_collabfilter_matrix,weighted_popularity_matrix)
final_ratings = np.round(final_ratings, 3)
final_ratings = pd.DataFrame(data=final_ratings)
print('')
print('shape of final matrix:', final_ratings.shape)

# save
final_ratings.to_csv('ratings.csv')
