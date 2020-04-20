# -*- coding: utf-8 -*-
"""
[Kaggle] NLP Real of Not - Bert 를 이용해서 분류

Kaggle url : https://www.kaggle.com/c/nlp-getting-started/overview
Model url : https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1

L-12 : hidden Layer 12개
H-768 : hidden size 768
A-12 : Attention Heads 12
"""

import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from RealOrNot.code import tokenization
from RealOrNot.code import dataClean
from RealOrNot.code import bertModel

root_path = os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename))
file_path = root_path + '\\RealOrNot'


def load_data(file_name) :
    return pd.read_csv( file_path +'\\dataset\\' + file_name)



train_data = load_data('train.csv')
test_data = load_data('test.csv')

print(train_data[:5])
print(test_data[:5])



cleanClass = dataClean.DataClean(train_data)
print(cleanClass.df)

x_data = np.array(cleanClass.df)
y_data = np.array(train_data['target'])


print(x_data[:5])

bert_layer = bertModel.bert_layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()






















