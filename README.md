This is a implementation of an implicit collaborative filtering where recommended products to user is done based on purchase history. As we can see from the data, there is no user rating or scores to consider it as an explicit data. So mixpanel data clearly shows type of implicit. 

Alternating Least Squares, popularly used implicit recommender technique, is a type of Matrix factorization to reduce the dimensionality of User/Product interaction. In other words we will find which products user has bought and which products the user never buy by interaction matrix. (Refer: http://yifanhu.net/PUB/cf.pdf)  we will fit our model using ALS implicit method from implicit library. Note: The implicit module requires Microsoft Visual C++ 14.0 to install and import.

There is an important concept of Dot product that we use for finding the recommended products. The dot product calculates the similarity between the items and the users. It is given by
                             **Score = User.〖Items〗^T**
This similarity scores helps us in recommending the products that are similar to the one which user purchased. 

# ALS Implicit recommendation  
We will import text file, convert it to dataframe and check for null values. Remove rows if null values are present. Let’s create a separate dataframe for Product ID and description to get information on products and need this for later use. 

Select columns such as 'product_id', 'quantity', 'customer_id' and drop all other columns. Convert all these columns to numerical types using cat and create a sparse matrix. The sparse matrix here contains both interactions and non-interactions between user and products.

Now train the data using ALS Model with iteration 25 (this can be increased based on the requirement) and alpha to 15. Alpha can be ranged from 15 to 40 depending upon the model performance, used in calculating confidence and regularization to avoid overfitting of data while training. The implicit ALS trains model much faster and gives User and Item vector from trained model. 

Two functions are created to get the purchased items by an user and recommended items to the same user (remember the dot product from user and item vector gives rating, based on the ratings the system recommend items) 

# AUROC: To check model performance

Splitting the sparse matrix into trainData and TestData. In TrainData we select interactions randomly and change the binary value to 0 and leave out these 0 interactions. In testData, we have all the interactions made by user-item and labelled it as 1 (Refer: https://www.slideshare.net/g33ktalk/dataengconf-building-a-music-recommender-system-from-scratch-with-spotify-data-team). We have train data, test data and the user index data where we set the binary value 0 in train data. From roc_curve we calculate the AUC value for predicted ratings (dot product of user_vecs and item_vecs) with the Testdata. 


References:
1) https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

2) http://www.salemmarafi.com/code/collaborative-filtering-with-python/

3) https://www.slideshare.net/g33ktalk/dataengconf-building-a-music-recommender-system-from-scratch-with-spotify-data-team

4) http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

5) https://github.com/benfred/implicit
