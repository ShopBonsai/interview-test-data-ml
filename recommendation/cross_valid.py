from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import SQLTransformer, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from metrics import RankingMetrics
import numpy as np
import os
import csv


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.OFF)


class ALSTrainFlow(object):

    def __init__(self, rating_path, user_path, book_path):
        if os.path.isfile(rating_path) is False:
            raise ValueError("rating_path is not valid path {}".format(rating_path))
        else:
            self.rating_path = rating_path

        if os.path.isfile(user_path) is False:
            raise ValueError("user_path is not valid path {}".format(user_path))
        else:
            self.user_path = user_path

        if os.path.isfile(book_path) is False:
            raise ValueError("book_path is not valid path {}".format(book_path))
        else:
            self.book_path = book_path

        self.sc = SparkContext()
        self.spark = SparkSession(self.sc)

        quiet_logs(self.sc)
        quiet_logs(self.spark)

        self.sc.setCheckpointDir('./checkpoint/')

    def read_data(self):

        # rdf is rating data frame
        self.rdf = self.spark.read.csv(path=self.rating_path, header=True,
                                       inferSchema=True, mode='DROPMALFORMED').drop('_c0')

        self.book_df = self.spark.read.csv(path=self.book_path, header=True,
                                           inferSchema=True, mode='DROPMALFORMED').select('bookISBN')

        self.user_df = self.spark.read.csv(path=self.user_path, header=True,
                                           inferSchema=True, mode='DROPMALFORMED').select('user')

        self.__show_rating_stat()

    def __show_rating_stat(self):
        num_ratings = self.rdf.count()
        num_users = self.rdf.select('user').distinct().count()
        num_items = self.rdf.select('bookId').distinct().count()

        sparsity = (num_ratings / float((num_users * num_items))) * 100

        print('Rating count : {}'.format(num_ratings))
        print('User count : {}'.format(num_users))
        print('Book count : {}'.format(num_items))
        print('Data Sparsity : {}%'.format(sparsity))

    def cross_validate_model(self, ranks, regParams, test_portion=0.1):
        if type(ranks) is not list or len(ranks) == 0:
            raise ValueError("ranks has to be non empty list but  "
                             "{} instead ".format(ranks))

        if type(regParams) is not list or len(regParams) == 0:
            raise ValueError("regParams has to be non empty list but  "
                             "{} instead ".format(regParams))

        if type(test_portion) is not float or (test_portion >= 1.0) or (test_portion <= 0.0):
            raise ValueError("test_portion has to have value 0.0 < test_partion < 1.0 but "
                             "{} instead ".format(test_portion))

        self.rdf = self.rdf.withColumn('rating', F.when(self.rdf.impression == 'checkout', 0.95).
                                       when(self.rdf.impression == 'add to cart', 0.8).
                                       when(self.rdf.impression == 'like', 0.7).
                                       when(self.rdf.impression == 'interact', 0.6).
                                       when(self.rdf.impression == 'view', 0.5).
                                       otherwise(0.25))

        self.rdf = self.index_book_ids()
        self.rdf = self.rdf.withColumn('ratingBin', F.when(self.rdf.impression == 'checkout', 1).otherwise(0))

        train, test = self.rdf.randomSplit([1 - test_portion, test_portion])

        als = ALS(maxIter=15, implicitPrefs=False, nonnegative=True,
                  userCol="user", itemCol="bookIdx", ratingCol="rating", rank=25)

        sqlTrans = SQLTransformer(
            statement="SELECT cast(prediction as DOUBLE ) as prediction, ratingBin  FROM __THIS__")

        pipeline = Pipeline(stages=[als, sqlTrans])

        paramGrid = ParamGridBuilder() \
            .addGrid(als.rank, ranks) \
            .addGrid(als.regParam, regParams) \
            .build()

        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                                                          labelCol="ratingBin",
                                                                          metricName="areaUnderROC"),
                                  numFolds=5)

        self.cvModel = crossval.fit(train)
        best_model = self.cvModel.bestModel.stages[0]

        train_params = [{p.name: v for p, v in m.items()} for m in self.cvModel.getEstimatorParamMaps()]
        auroc_scores = self.cvModel.avgMetrics

        self.save_metrics(train_params, auroc_scores)

        prediction = best_model.transform(test)

        rank_metrics = RankingMetrics(predictionCol='prediction', labelCol='ratingBin')
        self.avg_ndcg = rank_metrics.ndcgScore(prediction, by_whom='user')

        print ('NDCG score for best model is {} '.format(self.avg_ndcg))

        return best_model

    def index_book_ids(self):
        indexer = StringIndexer(inputCol="bookId", outputCol="bookIdx")
        return indexer.fit(self.rdf).transform(self.rdf)

    def save_metrics(self, params, metrics):
        result_list = []
        for par, score in zip(params, metrics):
            par['auroc_score'] = score
            result_list.append(par)

        result_list.sort(key=lambda x: x['auroc_score'], reverse=True)
        print('Best param set is {}'.format(result_list[0]))

        with open('cross_validation_metadata.csv', 'wb') as output_file:
            dict_writer = csv.DictWriter(output_file, result_list[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(result_list)

        self.best_params = result_list[0]

    def create_rating_matrix(self, model, path_to_save):

        ALSTrainFlow.__generate_rating_matrix(model=model, rating_df=self.rdf,
                                              user_sub_sample=self.user_df,
                                              book_sub_sample=self.book_df,
                                              path=path_to_save)


    @staticmethod
    def __generate_rating_matrix(model, rating_df, user_sub_sample, book_sub_sample, path):
        item_factors = model.itemFactors
        user_factors = model.userFactors

        bookIDs = rating_df.select('bookIdx', 'bookId').distinct()
        item_factors = item_factors.join(bookIDs, item_factors.id == bookIDs.bookIdx).drop('id', 'bookIdx')
        # Filter sub sample
        item_factors = item_factors.join(book_sub_sample, item_factors.bookId == book_sub_sample.bookISBN)

        item_factors_local = item_factors.rdd.map(lambda rec: [str(rec['bookId']), list(rec['features'])]).collect()
        # Filter user sub sample
        user_factors = user_factors.join(user_sub_sample, user_factors.id == user_sub_sample.user)

        user_factors_local = user_factors.rdd.map(lambda rec: [str(rec['id']), list(rec['features'])]).collect()

        user_factor_np = np.array([row[1] for row in user_factors_local])
        user_id_np = np.array([row[0] for row in user_factors_local])

        item_factor_np = np.array([row[1] for row in item_factors_local])
        item_id_np = np.array([row[0] for row in item_factors_local])

        rating_matrix = np.matmul(user_factor_np, item_factor_np.T)

        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['*'] + item_id_np.tolist())
            for i in range(len(user_id_np)):
                writer.writerow([user_id_np[i]] + rating_matrix[i].tolist())