"""
    日期：2019.5.1
    作者：jamke
    功能：预测燃油价格
    PS：
    分类问题：从类列表中选择一个类
    回归问题：从连续值中预测一个值

    # Use seaborn for pairplot
    !pip install -q seaborn
"""

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers

