# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-21'; 'last updated date: 2016-12-21'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import pandas as pd
from feature_encoder import FeatureEncoder, clean_text
from classifier import Classifier

FEATURES_TYPE = 'tf'
TOP_WORDS = 4
print('FEATURES_TYPE: %s' % FEATURES_TYPE)
print('TOP_WORDS: %d' % TOP_WORDS)

feature_encoder = FeatureEncoder(
    features_type=FEATURES_TYPE,
    columns=['title', 'content'],
)

classifier = Classifier(
    features_type=FEATURES_TYPE,

)


def load_data(file_name, header=0, encoding='utf8', index_col=None, converters=None):
    """ 加载数据

    :param file_name: str
        文件名
    :param header:
        是否有头文件
    :return:
    """
    data = pd.read_csv(file_name,
                       sep=',',
                       encoding=encoding,
                       header=header,
                       quoting=0,
                       converters=converters,
                       index_col=index_col,
                       )
    return data


def save_data(data, file_name, header=True, index=False):
    """ 保存数据

    :param index: bool
        是否保存索引
    :param data: pd.DataFrame()
    :param file_name: str
    :param header:
    :return:
    """
    data.to_csv(file_name, sep=',',
                quoting=1,
                quotechar='"',
                index=index,
                header=header,
                encoding='utf8')


def classifier_agent(dataset_names=None):
    print('分类器方法：%s' % classifier.features_type)
    print('TOP_WORDS: %d' % TOP_WORDS)

    if dataset_names is None:
        dataset_names = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    for name in dataset_names:
        assert name in ['biology', 'cooking', 'crypto', 'diy', 'robotics',
                        'travel'], 'dataset_names必须在[biology, cooking, crypto, diy, robotics, travel]中'
        train_df = load_data('dataset/train/%s.csv' % name)
        print('-' * 80)
        print('data name: %s , 数量：%d' % (name, len(train_df)))
        train_X = feature_encoder.fit_transform(
            train_df
        )
        train_y = train_df['tags']

        if FEATURES_TYPE in ['tf', 'tf-idf']:
            train_pred_y = classifier.predict(
                train_X, train_y, verbose=3,
                feature_names=feature_encoder.feature_names,
                top_words=TOP_WORDS
            )
        else:
            train_pred_y = classifier.predict(
                train_X, train_y, verbose=3,
            )


def test_predict():
    test_df = load_data('/home/jdwang/PycharmProjects/TransferLearning/dataset/test/test.csv')
    test_X = feature_encoder.fit_transform(
        test_df
    )
    test_pred_y = classifier.predict(
        test_X, None,
        verbose=1,
        feature_names=feature_encoder.feature_names,
        top_words=3
    )
    test_df['tags'] = test_pred_y
    save_data(
        test_df[['id', 'tags']],
        '/home/jdwang/PycharmProjects/TransferLearning/submit-tf.csv'
    )


# test_predict()