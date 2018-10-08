
# coding: utf-8

# In[1]:

'''Import Libraries'''

import csv
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import cross_validation as cv
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, confusion_matrix


# In[2]:

'''upload data file'''

data = pd.read_csv('F:/UsersData.csv', encoding='latin-1')


# In[3]:

'''estimate the missing age of user with the total mean value'''

data['age'].fillna(data['age'].mean(), inplace=True)
data['age'] = data['age'].astype('int')


# In[4]:

'''use label encoder to encode strings value to numeric value for the classification model'''

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
    
data['location']=le1.fit_transform(data['location'])
data['bookName']=le2.fit_transform(data['bookName'])
data['author']=le3.fit_transform(data['author'])
data['publisher']=le4.fit_transform(data['publisher'])


# In[5]:

'''split data into X and Y'''

X = data.iloc[:,0:9]
Y = data.iloc[:,9]


# In[6]:

'''split data into train and test'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[8]:

'''classifier model'''

model = xgb.XGBClassifier(base_score=0.5,colsample_bytree=0.8, gamma=0.05, learning_rate=0.001,max_depth=5
                          , min_child_weight=3, missing=None, n_estimators=1000,nthread=6, objective='multi:softmax'
                          ,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, random_state=42, silent=0,subsample=0.5)


# In[9]:

'''Fit data in the model'''

model.fit(X_train, y_train)


# In[10]:

'''make prediction for the test data'''

pred=model.predict(X_test)


# In[12]:

'''evaluate predictions'''
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[13]:

'''select other data attributes after prediction'''

recommendation=pd.DataFrame({'user':X_test['user'],'bookName':le2.inverse_transform(X_test['bookName']),'impression': pred })


# In[14]:

'''convert prediction column real responses'''

recommendation['impression'].replace( 1 ,'dislike',inplace=True)
recommendation['impression'].replace(2,'like',inplace=True)
recommendation['impression'].replace(3,'view',inplace=True)
recommendation['impression'].replace(4,'interact',inplace=True)
recommendation['impression'].replace(5,'add to cart',inplace=True)
recommendation['impression'].replace(6,'checkout',inplace=True)


# In[15]:

'''save output user recommendation file'''

recommendation.to_csv("F:/final_recomm.csv",index=False,encoding='utf-8')


# In[ ]:



