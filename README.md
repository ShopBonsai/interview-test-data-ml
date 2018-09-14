# Bonsai Interview Test
Welcome to the Machine Learning interview test for Shop Bonsai.

This interview test simulates a problem that is closely related to what you would be doing here at Shop Bonsai. 

### Scenario:
You joined as the new memeber of a small start-up team. Together we are building a new app to sell cool 3rd product 
products! So far, the sales team worked tirelessly and managed to acquire over 100 merchants who each have different 
brands and products offerings. The developers have made a ton of progress on the mobile app and there's a bunch of
user activity on the app. The next step is to optimize user conversion rates by offering new recommendations based on the analytics data. 

### Goal:
Your task is to recommend a book that a user is most likely to buy next using the user-event data provided. You can find the dataset here: https://www.dropbox.com/sh/uj3nsf66mtwm36q/AADLUNVShEZ0VI3DsLad6S4Ta?dl=0 

Note that your model should be feasible in a production environment - I.E. a complex, deeplearning model might outperform simpler models in terms of recommendation results, but it will be very slow to train and execute. In production, a recommender system will often make batch recommendations for all our current users, and should therefore be fast enough to warrant use. Aim to present a good balance between speed and evaluation metrics (see below). 

### Evaluation:
Your final submission should include all relevant code used, and an output csv file. The csv file should be a 
M x N matrix, where M is the number of users and N is the number of products, where each entry r<sub>ij</sub> 
represents the rating of product j for a given user i. A higher rating indicates that the user is more likely to 
purchase a product.

Your model will be evaluated by using [AUROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). You 
will be further evaluated on the quality of your coding style and model quality, methodology, and documentation.

High scorers will be contacted via email within a week of acknowledgement of PR submission.
Thank you and good luck for everyone who applied and submitted a PR.
