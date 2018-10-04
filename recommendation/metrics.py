import numpy as np
from pyspark.sql.types import FloatType, BooleanType
from pyspark.sql.functions import udf
from pyspark.sql.window import Window
import pyspark.sql.functions as F


class RankingMetrics(object):

    def __init__(self, labelCol, predictionCol):
        if type(labelCol) is not str:
            raise ValueError("labelCol must me str, but instead {}".format(type(labelCol)))

        if type(predictionCol) is not str:
            raise ValueError("predictionCol must me str, but instead {}".format(type(predictionCol)))

        self.labelCol = labelCol
        self.predictionCol = predictionCol

        """ NDCG score udf's"""
        self.calc_ndcg_udf = udf(lambda y: RankingMetrics.calc_ndcg(y), FloatType())
        self.ones_filter_udf = udf(lambda y: RankingMetrics.ones_filter(y), BooleanType())

    def ndcgScore(self, prediction_df, by_whom='user'):
        """
        :param prediction_df: dataframe which has labelCol, preditionCal, by_whom column.
        :param by_whom: parameter which describe by that column data have to be aggregated to calculate
         ndcg score. For instance 'user', in this case ndcg score will be calculated per user and then
         averaged over all users.
        :return: return averaged ndcg score.

        """

        if by_whom not in prediction_df.columns:
            raise BaseException('Column by_whom with value {} does not exists in DF'.format(by_whom))

        if self.labelCol not in prediction_df.columns:
            raise BaseException('Column labelCol with value {} does not exists in DF'.format(self.labelCol))

        if self.predictionCol not in prediction_df.columns:
            raise BaseException('Column predictionCol with value {} does not exists in DF'.format(self.predictionCol))

        w_user = Window.partitionBy(by_whom).orderBy(by_whom)

        ordered_pred = prediction_df.orderBy(by_whom, F.desc(self.predictionCol))

        pred_with_rank_list = ordered_pred.\
            select("*", F.collect_list(self.labelCol).over(w_user).alias("ordered_rank_list"))

        ranks_df = pred_with_rank_list.select(by_whom, 'ordered_rank_list').distinct()

        ndcg_df = ranks_df.withColumn('ndcg_score', self.calc_ndcg_udf(ranks_df.ordered_rank_list))

        # Filter all rank list that consists of only ones and zeros.

        ndcg_filtered_df = ndcg_df.where(self.ones_filter_udf(F.col('ordered_rank_list')) )
        ndcg_filtered_df = ndcg_filtered_df.where( F.col('ndcg_score') != 0)

        return ndcg_filtered_df.select(F.avg(F.col('ndcg_score'))).collect()

    @staticmethod
    def calc_ndcg(ranks):
        rank_list = list(ranks)
        k = len(rank_list)
        return float(RankingMetrics.ndcg_at_k(rank_list, k))


    @staticmethod
    def ones_filter(ar):
        return len(ar) != sum(ar)

    @staticmethod
    def dcg_at_k(r, k, method=0):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    @staticmethod
    def ndcg_at_k(r, k, method=0):
        dcg_max = RankingMetrics.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return RankingMetrics.dcg_at_k(r, k, method) / dcg_max


class TestRankingMetrics(object):

    def test_dcg_case1(self):
        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        assert RankingMetrics.dcg_at_k(r, 1) == 3.0

    def test_dcg_case2(self):
        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        assert RankingMetrics.dcg_at_k(r, 2) == 5.0

    def test_dcg_case3(self):
        r = []
        assert RankingMetrics.dcg_at_k(r, 2) == 0.0

    def test_ndcg_case1(self):
        r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        assert RankingMetrics.ndcg_at_k(r, 1) == 1.0

    def test_ndcg_case2(self):
        r = [1, 1, 1, 1]
        assert RankingMetrics.ndcg_at_k(r, 4) == 1.0

    def test_ndcg_case3(self):
        r = []
        assert RankingMetrics.ndcg_at_k(r, 1) == 0.0

