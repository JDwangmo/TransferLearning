# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-22'; 'last updated date: 2016-12-22'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import re
import numpy as np
import stop_words
from sklearn.feature_extraction import stop_words as sklearn_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# stop word list
STOP_WORDS = set(
    stop_words.get_stop_words('english')
    + list(sklearn_stopwords.ENGLISH_STOP_WORDS)
    + ['i', 'why', 'what', 'a']
)
print('the numbers of STOP WORDS: %d' % len(STOP_WORDS))


def clean_text(text):
    """清理文本

    :param text:
    :return:
    """
    # 全部转为小写
    text = text.lower()
    # 去掉html字符
    text = re.sub('<.*?>', ' ', text)
    # 去掉数字
    # text = re.sub('\d+', ' ', text)
    # 去掉非字母等字符
    text = re.sub('[^a-z\'-]', ' ', text)
    # 将多空白符全部 转为 单空格
    text = re.sub('\s+', ' ', text)
    # 去除stop words ,且长度大于1
    text = [item for item in text.strip().split(' ') if item not in STOP_WORDS and len(item) > 1]
    return text


class FeatureEncoder(object):
    """ 文本处理 特征工具类

    """

    def __init__(self,
                 features_type='raw_word',
                 columns=None
                 ):
        """参数初始化

        :type columns: list
            处理的字段，默认 是 [title, content]

        :param features_type: str
            选择何种 特征类型 ：
                - raw_word： 切分 成 词
                - tf-idf : tf-idf 编码
        """

        self.vectorizer = None
        self.feature_names = None

        if columns is None:
            columns = ['title', 'content']
        self.columns = columns
        self.features_type = features_type

    def segment_text_to_words(self, X):
        """清理文本并将文本切分成词
        :param X: list(str)
            字符串列表
        :return:
        """
        if 'title' in self.columns and 'content' in self.columns:
            features = [clean_text(item[0]) + clean_text(item[1]) for item in X[['title', 'content']].values]
        elif 'title' in self.columns:
            features = [clean_text(item) for item in X['title']]
        elif 'content' in self.columns:
            features = [clean_text(item) for item in X['content']]
        else:
            raise NotImplementedError
        return features

    def fit(self, train_X):
        if self.features_type == 'raw_word':
            return self
        elif self.features_type == 'tf':
            # 处理输入格式
            features = self.segment_text_to_words(train_X)
            features = [' '.join(item) for item in features]
            # fit tf
            self.vectorizer = CountVectorizer()
            self.vectorizer.fit(features)
            # 字典
            self.feature_names = np.asarray(self.vectorizer.get_feature_names())
            return self
        elif self.features_type == 'tf-idf':
            # 处理输入格式
            features = self.segment_text_to_words(train_X)
            features = [' '.join(item) for item in features]
            # fit tf-idf
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(features)
            # 字典
            self.feature_names = np.asarray(self.vectorizer.get_feature_names())
            return self
        else:
            raise NotImplementedError

    def transform(self, X):
        """

        :type X: pd.DataFrame()
            - 数据
        """
        # 处理输入格式
        features = self.segment_text_to_words(X)
        features = [' '.join(item) for item in features]

        if self.features_type == 'raw_word':
            return features
        elif self.features_type in['tf', 'tf-idf']:
            # transform tf-idf or tf
            return self.vectorizer.transform(features)
        else:
            raise NotImplementedError

    def fit_transform(self, train_X):
        return self.fit(train_X).transform(train_X)
