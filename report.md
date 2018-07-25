# Purchase History based Recommender System 

## Instructions

### Install Python dependencies
pip -r requirements.txt

### Run
python implicit_colab.py path_to_mixpanel_training_file


## Literature Review
There are several techniques to build a recommendation engine based on user-user similarity or item-item similarity or user-item similarity. We will go through the prominent ones.

### Content-based
Content-based recommendation systems are good to solve a cold-start problem but it requires that we should have enough information about the products (say genre, music composer, guitarists, lyrics etc in case of music) and we should also be aware what features (such as mood, melody, tempo etc) constitute the user's taste. Deriving these features becomes much difficult if you have a short shell-life inventory.

### Collaborative Filtering (CF)
Collaborative filtering based recomendation systems are built using matrix factorization of user-item ratings matrix. But they don't work well when there is no negative feedback as well as no direct information about user's likeliness for a product from interactions such as user click data or purchase history.

### Implicit Collaborative Filtering (Implicit CF)
In 2008, Yifan Hu et.al. published a [collaborative filtering method based on implicit feedback](https://dl.acm.org/citation.cfm?id=1511352) 
In explicit feedback such as ratings indicate user's preference for a product whereas implicit feedback represents just confidence. For example someone buying 4 coffees and 4 pizzas doesn't mean that user has equal preference for both items. In general people consume more coffee in a day than pizza. This number simply means that if user buys something in higher quanity he/she is going to buy that again but we don't know for what reasons.

### Item2Vec
Item2Vec is an extension to a technique in Natural Language Processing called Word2Vec. The hypothesis of Word2Vec is that set of words who appear in close vicnity in a sentence are closely related and such words, if represented in vectors, will have smaller distance amongst them whereas words who appear farther in a sentence will have higher distance between them. When we train this model with the existing literature, we can represent words into mathematical vectors. Once you have the mathematical representation of words, you can calculate the similarity between two words.

Similar analogy was applied for Item2Vec. If we consider a purchase event to be a sentence and each product as a word we can train such a model and represent item ids into vector and calculate item-item similarity. For an example if a laptop, monitor, desk speakers, wireless keyboard and mouse are frequently bought together so they are close to each other whereas if laptop and hats are not bought together frequently they are not similar.

### Bandit Algorithms
Implicit CF suffers from cold-start problem as well as with an ever changing inventory. Bandit Algorithms solves both of these problems. Another advantage of Bandit Algorithms is that they quickly adapt to feedback so it balances exploration and exploitation well. Bandit algorithms are reinforcement learning techniques. Every time an item is recommended, it gets a reward, say +1 for correct prediction and -1 for incorrect prediction. It learns to maximize the reward i.e. to keep predicting the correct recommendation.

Contextual or multi-arm bandit algorithms are even better for practical use cases where a recommendation is based on several features. The algorithm learns which feature is more important based on feedback and eventually the best features are picked. 

Baynote and Rich Relevance, top two e-commerce product recommendation_engine-as-a-service companies, use bandit algorithms.
They are well-suited for domains like news where the number of candidate items is near-constant and almost all items have little past feedback because articles go stale very quickly. In contrast, domains such as movies or music have thousands of candidate items to recommend, thus making application of contextual bandits a non-trivial task due to the curse of dimensionality.

## Data Analysis
The mixpanel data file has purchase events which include both purchases and returns. Each such event has product id, customer id, timestamp, quantity, unit price and product description. Returned items have negative quantity values. There are no null values for any of the fields, so null values replacement/cleaning was not required. There are few users who returned certain products but there is purchase history for these customer-product pairs. I have removed such events from the dataset.


## Algorithm Selection
In this dataset, we have purchase history as an implicit feedback from users unlike ratings which is explicit feedback. So I used Implicit CF to train the recommendation system.

## Feature Selection
We have certain feedbacks from customers:

- price: for a similar product customer's have a certain preference. some of them value the brand name whereas others prefer price/value factor.
- quantity: higher the quantity higher would be the likeness for the product
- purchase/return event: returned item shows dislike whereas purchased but not returned shows likeness

I have not included unit price as a feature because it is a latent factor in user's purchase behaviour moreover price alone doesn't represent a complete customer profile.
Similarily, purchase/return factor can be taken included if we take total quanity of an item bought buy a customer. Higher the number higher would be the likeness of the product by that customer. So total quantity can be considered a good feature. However, I feel purchase frequency (number of items bought per month/year) is a better indicator than total quantity because it shows user is a frequent visitor to website/store. The chances of up-sell are higher if a user visits store/website more often.


## Data Transformation
To calculate purchase frequency, total quantity of an item purchased by a user was divided by the time difference (in days) of last time and first time, this item was bought by this user. There were an issue in this approach. Time difference was 0 for:
- items bought only once
- items bought more than once but just on one day

To solve these edge cases, time difference was takes as the maximum time difference across all purchase events i.e. we assumed that during all the purchase history we have such user-item purchase interaction happened once.

## Training & Validation
We used an open source Python implementation of paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) called [implicit](https://github.com/benfred/implicit).
For training, we split the dataset into train and test (say a 70-30 split). Default parameters are used for the initial training.

Evaluation is done by calculating area under ROC curve.


## Hyperparameter Tuning
After running a number of experiments with different values of number-of-latent-factors and regularization-parameter, it was concluded that model converges (i.e. loss becomes steady) in 40-50 iterations. So, for further experiments, value of iterations was fixed to 40 in order to reduce number of experiments for parameters tuning.
For hyperparameter tuning, we ran experiments for 
- factors = [5, 10, 20, 40, 60, 80, 100, 150, 180] and
- regularization = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

Finally, we picked the hyperparameters which were giving highest AUROC and precision values (in given order)
File: implicit_colab.log contains the logs produced during hyperparameter training.

For the given dataset, these values are - factors = 20 and regularization = 0.001

## Results
Best set of results achieved is for factors=20 and regularization=0.001
AUROC = 0.85 and precision at 10-fold = 0.1

### General Observations
- Overfitting
  Precision increase as we increase number of latent factors. There can be 2 cases here:
  - overfitting: model is overfitting on the training data
  - learning: we need more factors because the user-item relationship actually contains higher number of latent factors
  Since we can also observe that AUROC increases from factors=5 to factors=20 and later decreases for factors>20 i.e. False Positives are increasing for factors>20. This means that model is overfitting for factors>20.

- Regularization
  Higher the regularization, lower will be the overfitting. Model is overfitting for higher values of factors that's where regularization is playing a role. We can see that for factors>100, when we increase the regularization parameter AUROC increases.

## Future Work
CF suffers from the cold-start problem i.e. we can't generate recommendations for users who don't have enough purchase history. To resolve this we can:

- suggest top products in a category for users with zero purchase history
- recommend items based on item-item similarity for users with less than a threshold (say 5 or 10) purchase history

For item-item similarity we can also use Item2Vec. 
