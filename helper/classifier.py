# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-22'; 'last updated date: 2016-12-22'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np


def get_metrics(true_y, pred_y):
    precision_score_list, recall_score_list, f1_score_list = [], [], []
    for real, pred in zip(true_y, pred_y):
        real = real.split()
        pred = pred.split()

        TP = sum([item in real for item in pred])
        NP = sum([item not in real for item in pred])
        precision_score = TP / float(TP + NP)

        FN = sum([item not in pred for item in real])
        recall_score = TP / float(TP + FN)
        if precision_score + recall_score == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)

        precision_score_list.append(precision_score)
        recall_score_list.append(recall_score)
        f1_score_list.append(f1_score)

    print('precision_score：%f,recall_score：%f,f1_score：%f' % (
        np.mean(precision_score_list), np.mean(recall_score_list), np.mean(f1_score_list)
    ))


class Classifier(object):
    """ 分类器

    """

    def __init__(self,
                 features_type='tf',
                 ):
        """


        :type features_type: str
            特征类型：
                - tf： 词频统计方法
        """
        self.features_type = features_type

    def fit(self, train_X, train_y):
        if self.features_type in ['tf', 'tf-idf']:
            print('tf不需要fit！')
            return self

    def predict(self, test_X, test_y, verbose=3, **args):
        """ 预测结果

        :param verbose: int
            数值越大，越详细
        :param test_X: array-like
            e.g. [['1','1','2','2','3']]
        :param test_y:
        :type args: dict
            - tf-idf_feature_names: tf-idf 词汇字典/特征名
            - top_words: 取分数 top-n 个结果
        :return:
        """
        feature_names = args.get('feature_names', None)
        assert feature_names is not None, '没有设置feature_names！'
        print('字典大小： %d' % (len(feature_names)))
        top_words = args.get('top_words', 2)

        if self.features_type in ['tf', 'tf-idf']:
            result = []
            for item in test_X:
                r = ' '.join(feature_names[np.argsort(item.toarray().flatten())[-top_words:]])
                result.append(r)
            if verbose > 2:
                get_metrics(test_y, result)
            return result
        else:
            raise NotImplementedError


def test_tf():
    classifier = Classifier(features_type='tf')
    classifier.predict([['1', '1', '2', '2', '3'], ['1', '1', '3', '3', '3']], ['1 2', '3'])
