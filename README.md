# Bonsai Interview Test

Welcome to the Machine Learning interview test for Shop Bonsai.

This interview test simulates a problem that is closely related to what you would be doing here at Shop Bonsai. 

### Scenario:

You joined as the new memeber of a small start-up team. Together we are building a new app to sell cool 3rd product 
products! So far, the sales team worked tirelessly and managed to acquire over 50 merchants who each have different 
brands and products offerings. The developers have made a ton of progress on the mobile app and there's a bunch of
user activity on the app. The analytics platform Mixpanel is used to collect all the user event data.
The next step is to optimize user conversion rates by offering new recommendations based on the analytics data.

### Goal:
Your task is to recommend a product that a user is most likely to buy next using the purchase history provided.

For the purpose of this interview test, we have provided mock data on customer purchase history from an e-commerce 
retail company. The 'Purchased Product' events were queried through Mixpanel's API and is exported into the file 
`training_mixpanel.txt`, as provided, in JSON format. Each event describes a singular product purchased by a 
particular user, with  descriptive attributes of the product (e.g., quantity, unit price). Transactions purchasing 
multiple products is denoted by `invoice_no`.


### Evaluation:
Your final submission should include all relevant code used, and an output csv file. The csv file should be a 
M x N matrix, where M is the number of users and N is the number of products, where each entry r<sub>ij</sub> 
represents the rating of product j for a given user i. A higher rating indicates that the user is more likely to 
purchase a product.

Your model will be evaluated by using [AUROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). You 
will be further evaluated on the quality of your coding style and model quality, methodology, and documentation.

High scorers will be contacted via email within a week of acknowledgement of PR submission.
Thank you and good luck for everyone who applied and submitted a PR.
