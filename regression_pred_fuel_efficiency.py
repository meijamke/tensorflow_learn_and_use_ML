"""
    日期：2019.5.1
    作者：jamke
    功能：预测燃油价格
    PS：
    分类问题：从类列表中选择一个类
    回归问题：从连续值中预测一个值

    # Use seaborn for pairplot
    pip install -q seaborn
"""

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers

"""
    下载数据集
    使用pandas导入
"""
dataset_path = keras.utils.get_file('auto-mpg.data',
                                    'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values='?', comment='\t',
                          sep='', skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

"""
    预处理数据
"""
dataset.isna().sum()
dataset = dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

print(dataset.tail())

"""
    切分数据集
    训练集+测试集
"""
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
