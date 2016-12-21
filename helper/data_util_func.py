# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-21'; 'last updated date: 2016-12-21'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import pandas as pd
import numpy as np


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
                       # converters=converters,
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


def show_df_info(data=None):
    """显示数据某个属性的详情：

    :param data: pd.DataFrame()
        数据框
    """
    pass
    print(data.head())
    print(data.info())
