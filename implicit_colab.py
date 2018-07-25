# coding: utf-8

# import Python libraries

import matplotlib
import matplotlib.pyplot as plt

import json
import time
import sys
import logging
import logging.handlers
import numpy as np
import pandas as pd
from sklearn import metrics
import scipy
from implicit.evaluation import train_test_split, precision_at_k, mean_average_precision_at_k
from implicit.als import AlternatingLeastSquares

# create logger
logger = logging.getLogger('implicit_colab_filtering')
logger.setLevel(logging.DEBUG)

# create rotating file handler which logs even debug messages
rfh = logging.handlers.RotatingFileHandler('implicit_colab.log', 
                                            maxBytes=2048*1024, backupCount=5)
rfh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
rfh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(rfh)
logger.addHandler(ch)

def calc_purchase_frequency(df):
    """Calculates purchase frequency using given data
    and returns it in a new dataframe
    """
    # group columns by customer_id, then product_id
    g = df.groupby(['customer_id', 'product_id'])
    # add quantities of each user, item pair
    # since quantity is negative for returned items
    # sum would be purchased - returned
    final_data = g[['quantity']].sum()

    # We will drop the events where total quantity is not positive.
    # These are the cases where we have record for returned items
    # but not when these items were purchased.
    final_data = final_data[final_data.quantity > 0]

    # For each user, item pair find time difference between last and first transaction.
    final_data['time_delta'] = g[['invoice_date']].max() - g[['invoice_date']].min()

    # Convert interval from time to just days
    final_data['days_delta'] = final_data['time_delta'].dt.days

    # Value of days interval will be zero when single or multiple transactions 
    # for a given user-item pair have occurred only on 1 day through purchase history
    # We will replace these zero values by total time difference (in days)
    # throughout purchase history
    tdelta_max = final_data['time_delta'].max().days
    final_data.loc[final_data['days_delta']==0, 'days_delta'] = tdelta_max

    # Calculate purchase frequency per user, item pair
    final_data['purchase_freq'] = final_data['quantity'] / final_data['days_delta']

    # Drop rows with NA or Null values
    final_data.dropna(inplace=True)
    return final_data


def get_user_item_matrix(final_data):
    """Converts user-item purchase data to user-item sparse matrix
    """
    users = list(final_data.index.get_level_values('customer_id').unique())
    items = list(final_data.index.get_level_values('product_id').unique())

    # reset index 
    final_data.reset_index(level=['customer_id', 'product_id'], inplace=True)

    # 
    user_indices = final_data['customer_id'].astype('category').cat.codes
    item_indices = final_data['product_id'].astype('category').cat.codes
    purchase_frequency = list(final_data['purchase_freq'])

    # create a sparse matrix for user-item purchase frequency
    user_items = scipy.sparse.csr_matrix((purchase_frequency, (user_indices, item_indices)),
                                         shape=(len(users), len(items)))
    return user_items


def normalise_data(data):
    """Normalizes data such that positive examples (non-zero values) and 
    negative examples (zero values) have almost equal weight.
    Ref: https://github.com/benfred/implicit/issues/74
    """
    alpha = (data.shape[0] * data.shape[1] - data.nnz) / sum(data.data)
    data = data*alpha
    return data


def train(model, train_data):
    """train ALS model
    """
    t1 = time.time()
    model.fit(train_data)
    t2 = time.time()

    return model, t2-t1


def calc_precision(model, train, val):
    """Calculates precision and mean average precision at top-K
    """
    p = precision_at_k(model, train.T.tocsr(), val.T.tocsr(), K=10, num_threads=2)
    ma_p = mean_average_precision_at_k(model, train.T.tocsr(), val.T.tocsr(), K=10, num_threads=2)
    return p, ma_p


def calc_auroc(model, val_data):
    """calculates are under ROC curve
    val: validation data in form of sparse matrix 
    model: model trained on training data
    """

    t1 = time.time()
    # convert validation data to numpy array
    val_dense = val_data.todense().transpose()
    val_array = np.asarray(val_dense)
    # Convert weights to binary values (1=purchased, 0=not-purchased)
    val_array[val_array!=0] = 1


    # Scipy's roc_curve funtion expects 
    # ground truth values to be labels such as 0, 1
    # and predictions to be classification probabilities 
    # So we will create variables following this format

    # convert binary values from float to int
    actual = np.array(val_array, dtype=int)

    # generate predictions
    user_vectors = model.user_factors
    item_vectors = model.item_factors
    predictions = np.dot(user_vectors, item_vectors.transpose())
    # predictions = np.zeros(val_array.shape, dtype=float)
    # for i in range(val_array.shape[0]):
    #     # we will skip generating predictions for users who don't  
    #     # have any purchased items in validation dataset
    #     if np.sum(val_array[i, :]) != 0:
    #         predictions[i, :] = np.dot(user_vectors[i, :], item_vectors.transpose())

    # true positive rate and false positive rate at various thresholds
    fpr, tpr, ths = metrics.roc_curve(actual.ravel(), predictions.ravel())
    # calculate area under the curve using TPR and FPR
    auroc = metrics.auc(fpr, tpr)

    t2 = time.time()
    return tpr, fpr, auroc, t2-t1


def plot_roc(tpr, fpr, auroc, filename=None):
    """Plot Receiver Operating Characteristic curve
    """
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.2f}'.format(auroc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close('all')


if __name__ == '__main__':

    HYPERPARAMETER_TUNING = True

    training_filepath = sys.argv[1]    
    logger.info('Reading data')
    # Read training data from file
    logger.debug('reading data file from disk')
    with open(training_filepath) as f:
        data = json.loads(f.read())

    # convert all purchase events from json to a Pandas dataframe
    logger.debug('creating DataFrame from json object')
    df = pd.DataFrame(data=[event['properties'] for event in data])
    logger.debug("Dataframe information")
    logger.debug('{} {}'.format('\n', df.info()))

    # invoice_date is stored as object (i.e. string).
    # We are going to use datetime later, so let's convert it from string 
    # to datetime objects. This operation is quite slow and takes time.
    logger.debug('converting date strings to datetime - start')
    df['invoice_date'] = df['invoice_date'].apply(pd.to_datetime)
    logger.debug('converting date strings to datetime - end')

    logger.debug("Dataframe Description")
    logger.debug('{} {}'.format('\n', df.describe()))
    logger.debug("Dataframe top 5 rows")
    logger.debug('{} {}'.format('\n', df.head()))

    # Minimum value of quantity is negative which indicates 
    # that there are returned orders too.
    # Later on we will see that returned orders having negative quantities
    # is helpful in our calculations.
    logger.info('total number of return transactions: {}'.format( 
          df.quantity[df.quantity < 0].count()))
    logger.info('total number of customer visits when products were returned:'
                '{}'.format(df.invoice_no[df.quantity < 0].unique().shape[0]))
                                   

    # ========= data processing =========
    logger.info('Transforming data')
    # create a new data frame which has total quantity purchased, time interval
    # between first & last purchase, and purchase frequency 
    final_data = calc_purchase_frequency(df)


    # create a user-item sparse matrix with purchase frequency as weights
    user_items = get_user_item_matrix(final_data)

    # release memory
    del(df)
    del(final_data)

    # ALS implemented in implicit module accepts the item-user matrix
    item_users = user_items.transpose()


    # ========= training and validation =========
    # Normalise data before matrix factorization using ALS
    logger.info('Training and Validation')
    user_items = normalise_data(user_items)
    item_users = normalise_data(item_users)

    # split data for training and validation
    train_data, val_data = train_test_split(item_users)

    # initialize ALS model object
    model = AlternatingLeastSquares(factors=20, regularization=0.01, iterations=40,
                                    calculate_training_loss=True)
    # train model on training data
    model, ttrain = train(model, train_data)

    # evaluate True Positive Rate, False Positive Rate and AUROC on validation dataset
    tpr, fpr, auroc, teval = calc_auroc(model, val_data)

    p, ma_p = calc_precision(model, train_data, val_data)
    logger.info('training time: {:0.1f}s evaluation time: {:0.1f}s'.format(ttrain, teval))
    logger.info('AUROC: {:0.2f}, precision: {:0.3f}, '
          'mean average precision: {:0.3f}'.format(auroc, p, ma_p))

    # # Plot ROC curve
    # plot_roc(tpr, fpr, auroc)


    # ========= hyperparameter tuning =========
    # It was observed that after 35-40 iterations, loss stops decreasing 
    # at a much slower rate which means that model has converged.
    # So for rest of the hyperparameters tuning, we are going to use iterations=40

    # This is a time taking process. Set HYPERPARAMETER_TUNING to True 
    # if you want to run this again.
    logger.info('Hyperparameter Tuning')
    if HYPERPARAMETER_TUNING:
        factors_range = [5, 10, 20, 40, 60, 80, 100, 150, 180]
        reg_range = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

        # variable to store values of factors, regularization, precision, 
        # mean avg precision and AUROC
        data = []

        # Initialize plot
        num_rows = (len(factors_range)+1)/2
        # plt.tight_layout()
        plt.figure(figsize=(6*4, 6*num_rows))
        plt.title('Receiver Operating Characteristic')

        # iterate over factors list
        for i, factors in enumerate(factors_range):
            # Initialize subplot
            plt.subplot(num_rows, 2, i+1)
            # plt.title('factors={}'.format(factors), fontsize=11)
            legends = []
            # plot baseline with AUROC=0.5
            plt.plot([0,1],[0,1],'r--')
            legends.append('auroc=0.5')

            # iterate over regularization range
            for reg in reg_range:
                # intialize model
                model = AlternatingLeastSquares(factors=factors,
                                                regularization=reg,
                                                iterations=40, 
                                                calculate_training_loss=True)
                print('\n')
                train(model, train_data)

                # evaluate model
                p, ma_p = calc_precision(model, train_data, val_data)
                tpr, fpr, auroc, teval = calc_auroc(model, val_data)

                # to save each subplot
                # plot_filename = '{}_{:2.3f}_{:0.2f}.png'.format(factors, reg, auroc)
                # plot_roc(tpr, fpr, auroc, filename=plot_filename)

                # plot ROC curve
                plt.plot(fpr, tpr)
                legends.append('factors={}, r={:0.3f}, auroc={:0.2f}, p={:0.3f}, '
                               'map={:0.3f}'.format(factors, reg, auroc, p, ma_p))

                # collect data for this iteration
                data.append([factors, reg, p, ma_p, auroc])

                logger.info('factors: {}, regularization: {}, AUROC: {:0.2f}, '
                      'p: {:0.3f}, map: {:0.3f}'.format(factors, reg, auroc, p, ma_p))

            plt.xlim([-0.1,1.2])
            plt.ylim([-0.1,1.2])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(legends, loc='lower right', fontsize='x-small')

        # save plot on disk
        plt.savefig('hyperparam_tuning.png')
        plt.close()
        # plt.show()

        # create a dataframe
        columns = ['factors', 'regularization', 'precision', 'mean_avg_precision', 'area_under_roc']
        df_results = pd.DataFrame(data=data, columns=columns)

        # sort dataframe
        df_results_sorted = df_results.sort_values(['area_under_roc','precision'], ascending=False)
        logger.info('{} {}'.format('\n', df_results_sorted.head()))
        # save hyperparameter tuning results to disk
        df_results_sorted.to_csv('hyperparam_tuning_results.csv', sep=',', encoding='utf-8')

        factors_best = df_results_sorted['factors'][0]
        regularization_best = df_results_sorted['regularization'][0]
    # values from previous runs
    else:
        factors_best = 20
        regularization_best = 0.001


    # ========= generating results =========
    # Based on results from hyperparameter tuning, we are going to pick parameters
    # where first AUROC is maximum, then precision is maximum
    # Train model on training + validation dataset
    logger.info('Generating final results based on best set of hyperparams')
    model = AlternatingLeastSquares(factors=factors_best,
                                    regularization=regularization_best,
                                    iterations=40,
                                    calculate_training_loss=True)

    logger.info('training model on all data')
    model, ttrain = train(model, item_users)

    user_vectors = model.user_factors
    item_vectors = model.item_factors

    # user-item rating matrix
    logger.info('calculating user-items ratings matrix')
    user_item_rating = np.dot(user_vectors, item_vectors.transpose())

    # save user-item ratings matrix to disk
    logger.info('saving user-item ratings to disk')
    np.savetxt('user-item-ratings.csv.gz', user_item_rating, fmt='%.4e', delimiter=',', encoding='utf-8')
