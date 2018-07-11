import json
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit 
import random
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

json_data=open('training_mixpanel.txt').read()
raw_data = json.loads(json_data)
raw_data =  pd.DataFrame((d['properties'] for d in raw_data))

# Gives null values and data types inforamtion of data 
rawdata_info=pd.DataFrame(raw_data.dtypes).T.rename(index={0:'column type'})
rawdata_info=rawdata_info.append(pd.DataFrame(raw_data.isnull().sum()).T.rename(index={0:'null values (numbers)'}))
rawdata_info=rawdata_info.append(pd.DataFrame(raw_data.isnull().sum()/raw_data.shape[0]*100).T.rename(index={0:'null values (%)'}))

## Product description 
product_info = (raw_data[['product_id', 'description']]).drop_duplicates() 
product_info['product_id'] = product_info['product_id'].astype(str)

data = raw_data[['product_id', 'quantity', 'customer_id']] # Selecting the required columns 
data = data.groupby(['customer_id', 'product_id']).sum().reset_index() # grouping and summing to check how many products were totally brought by each user
new_data = data[data.quantity > 0]  # Select User ID only if the quantity is more than 0

users = list(np.sort(new_data['customer_id'].unique())) # Get our unique User ID
products = list(new_data['product_id'].unique()) # Get unique products 
quantity = list(new_data['quantity']) 
rows = new_data['customer_id'].astype('category', categories = users).cat.codes 
cols = new_data['product_id'].astype('category', categories = products).cat.codes 
sparse_data = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(users), len(products)))

## training the model
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=25)
alpha_val = 15 # it can be ranged from 15 to 40
data_conf = (sparse_data* alpha_val).astype('double')
model.fit(data_conf)

user_vecs = model.item_factors  ## user  vectors 
item_vecs = model.user_factors  ##  item  vectors

## AUROC Test
def split_data(sparse_data):   
    trainData = sparse_data.copy() # creating a train data 
    Interactions = trainData.nonzero() #data where an interaction exists
    Interactions_data = list(zip(Interactions[0], Interactions[1])) #user,item index into a list
    random.seed(0)
    exp = int(np.ceil(0.2*len(Interactions_data))) 
    samples = random.sample(Interactions_data, exp) # Sample a random number of user-item pairs without replacement
    user = [index[0] for index in samples] #user row indices
    item = [index[1] for index in samples] #item column indices
    trainData[user, item] = 0 # Assign user-item pairs to zero
    trainData.eliminate_zeros() # eliminate zeros interactions to make sparse matrix dense  
    testData = sparse_data.copy()  
    testData[testData != 0] = 1 # change user interaction value to binary data 1 and others will have 0 value
    return trainData, testData, list(set(user))  

trainData, testData, user_index_test = split_data(sparse_data)

def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)   

def auc(trainData,user_index_test, vectors, testData):    
    aucs = [] 
    item_vecs = vectors[1]
    for user in user_index_test: 
        train = trainData[user,:].toarray().reshape(-1) #training set row
        interaction_zero = np.where(train == 0) # zero interaction 
        user_vec = vectors[0][user,:]
        prediction = user_vec.dot(item_vecs).toarray()[0,interaction_zero].reshape(-1) #taking dot vector to create recommendation
        actual = testData[user,:].toarray()[0,interaction_zero].reshape(-1) 
        aucs.append(auc_score(prediction, actual))
    return round(float(np.mean(aucs)),3)
  
auc(trainData, user_index_test,[sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], testData)


users1 = np.array(users) # Array User ID 
products1 = np.array(products)  

def consumed_product(customer_id, sparse_data, users1,products1,product_info):
    customer_index = np.where(users1 == customer_id)[0][0] 
    purchased_index = sparse_data[customer_index,:].nonzero()[1] 
    prod = products1[purchased_index] # Get consumed products
    return product_info.loc[product_info.product_id.isin(prod)]

def recommended_products(customer_id, sparse_data, user_vecs, item_vecs, users1, products1, product_info, num_items = 10):        
    customer_index = np.where(users1 == customer_id)[0][0] 
    interaction = sparse_data[customer_index,:].toarray() # Get Interaction of User and Product
    interaction = interaction.reshape(-1) + 1 # no recommend items the user has consumed. So let's set them all to 0 and the unknowns to 1.
    interaction[interaction > 1] = 0 # Make everything already purchased zero  
    recommend_vector = user_vecs[customer_index,:].dot(item_vecs.T) # Dot product for creating recommendation  
    min_max = MinMaxScaler()
    recommend_vector_scaled = min_max.fit_transform(recommend_vector.reshape(-1,1))[:,0] 
    recommended = interaction*recommend_vector_scaled #Rating matrix
    product_idx = np.argsort(recommended)[::-1][:num_items] # Sort the indices of the items into order 
    recommend_list = [] # start empty list to store items along with product ID 
    recommend_items = [] # stores items without product ID
    for index in product_idx:
        cd = products1[index]
        recommend_list.append([cd,product_info.description.loc[product_info.product_id == cd].iloc[0]]) 
        recommend_items.append([product_info.description.loc[product_info.product_id == cd].iloc[0]]) 
    return recommend_list,recommend_items,recommended

#pr  = consumed_product(12347,sparse_data, users1, products1,product_info)
#re = recommended_products(12347, sparse_data, user_vecs, item_vecs, users1, products1, product_info, num_items = 10)



## Recommended Item dataframe and Rating dataframe  (This might be slow)
final_data = pd.DataFrame()
final_data['User ID'] = users1
final_data['Purchased'] = final_data.apply(lambda x:(consumed_product(x['User ID'], sparse_data, users1,products1,product_info).iloc[:,1:2]).to_dict(),axis=1)
final_data['Recommended'] = final_data.apply(lambda x: recommended_products(x['User ID'], 
          sparse_data, user_vecs, item_vecs, users1, products1, product_info, num_items = 10),axis=1).astype(str)
final_data.to_csv('finaldata.csv')


final_ratings =  pd.DataFrame()
final_ratings.insert(loc=0, column='User ID', value= users1)
final_ratings['Ratings'] = final_ratings.apply(lambda x: recommended_products(x['User ID'], 
          sparse_data, user_vecs, item_vecs, users1, products1, product_info, num_items = 10)[2],axis=1)
#final_ratings = final_ratings.join(pd.DataFrame(final_ratings['Ratings'].values.tolist()))
final_ratings = final_ratings.join(final_ratings['Ratings'].apply(pd.Series))
del final_ratings['Ratings']
colname_prod = copy.deepcopy(products)
colname_prod.insert(0,'User ID')
final_ratings.columns = colname_prod
final_ratings.to_csv('final_ratings.csv')




