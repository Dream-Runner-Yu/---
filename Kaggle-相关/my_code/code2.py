

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# cross_val_score()

train = pd.read_csv("../data/data/train.csv")
test = pd.read_csv("../data/data/test.csv")
merchant = pd.read_csv('../data/data/merchants.csv')

print(merchant.info())
print(merchant.describe())
print(merchant.head())

print(merchant.columns)



# print(train.shape)
# print(test.shape)
#
# print(train.head())
# print(train.info())

# 数据集的质量分析
# print(train['card_id'].nunique() == train.shape[0])
# print(test['card_id'].nunique() == test.shape[0] )


