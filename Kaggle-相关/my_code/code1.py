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

print(train.shape)
print(test.shape)

print(train.head())
print(train.info())

# 数据集的质量分析
print(train['card_id'].nunique() == train.shape[0])
print(test['card_id'].nunique() == test.shape[0] )


print(train.isnull().sum())
print(test.isnull().sum())

# 异常值检测
print(train.describe())

import seaborn as sns
import matplotlib.pyplot as plt

# print(train.columns)
# sns.set()
# sns.histplot(train['target'], kde=True )
# plt.show()

print( (train['target'] < -30 ).sum() )

statistics = train['target'].describe()
print(statistics.loc['mean'],statistics.loc['std'])

# 规律一致性分析
# 单变量分析
# 特征分布
features = [ 'first_active_month', 'feature_1', 'feature_2', 'feature_3' ]
train_count = train.shape[0]
test_count = test.shape[0]
( train['first_active_month'].value_counts().sort_index() / train_count ).plot()
plt.show()

for feature in features:
    (train[feature].value_counts().sort_index() / train_count).plot()
    (test[feature].value_counts().sort_index() / test_count ).plot()
    plt.show()

# 多变量联合分布

def combine_feature(df):
    cols = df.columns
    feature1 = df[cols[0]].astype(str).values.tolist()
    feature2 = df[cols[1]].astype(str).values.tolist()
    return pd.Series([feature1[i]+'&'+feature2[i] for i in range(df.shape[0])])

cols = [features[0], features[1]]
train_com = combine_feature(train[cols])

train_dis = train_com.value_counts().sort_index()/train_count
test_dis = combine_feature(test[cols]).value_counts().sort_index()/test_count


# 创建新的index
index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates().sort_values()

# 对缺失值填补为0
(index_dis.map(train_dis).fillna(0)).plot()
(index_dis.map(train_dis).fillna(0)).plot()

# 绘图
plt.legend(['train','test'])
plt.xlabel('&'.join(cols))
plt.ylabel('ratio')
plt.show()

n = len(features)
for i in range(n-1):
    for j in range(i+1, n):
        cols = [features[i], features[j]]
        print(cols)
        train_dis = combine_feature(train[cols]).value_counts().sort_index()/train_count
        test_dis = combine_feature(test[cols]).value_counts().sort_index()/test_count
        index_dis = pd.Series(train_dis.index.tolist() + test_dis.index.tolist()).drop_duplicates().sort_values()
        (index_dis.map(train_dis).fillna(0)).plot()
        (index_dis.map(train_dis).fillna(0)).plot()
        plt.legend(['train','test'])
        plt.xlabel('&'.join(cols))
        plt.ylabel('ratio')
        plt.show()

