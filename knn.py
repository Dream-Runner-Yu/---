import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

print(iris)

print("-----")

df = pd.DataFrame(iris.data, columns= iris.feature_names )

df['class'] = iris.target
# print(df.head(10))
print(df.describe())

x = iris.data 
y = iris.target.reshape(-1,1)
# print(x.shape,y.shape)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify= y)
print(x_train.shape,x_test.shape)
print(x_test.shape,y_test.shape)

# k近邻核心实现
def l1_distance(a,b):
    return np.sum(np.abs(a-b),axis=1)

def l2_distance(a,b):
    return np.sqrt(np.sum((a-b)**2, axis= 1))

class kNN(object):
    # 初始化方法
    def __init__(self,n_neighbors = 1, dist_func = l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func
    
    # 模型预测
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
    
    # 模型训练
    def predict(self,x):
        y_pred = np.zeros((x.shape[0],1), dtype = self.y_train.dtype)
        # 遍历输入的数据
        for i, x_test in enumerate(x):
        # 计算距离
            distance = self.dist_func(self.x_train,x_test)
        # 排序并取出索引值
            nn_index = np.argsort(distance)

        # 选取最近的k个点统计类别，
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()

        # 选择频率最高的类别
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred    
    
# 测试
# 定义一个实例
# knn = kNN(n_neighbors=3)
# knn.fit(x_train,y_train)
# y_pred = knn.predict(x_test)
# #     求出预测准确率
# accuracy = accuracy_score(y_test,y_pred) * 100
# print("预测的准确率：", accuracy)
#

knn = kNN(n_neighbors=3)
knn.fit(x_train,y_train)

result_list = []
for p in [1,2]:
    # 考虑距离
    knn.dist_func = l1_distance if p == 1 else l2_distance
    # 考虑 k 值
    for k in range(1,10,2):
        knn.n_neighbors = k
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        result_list.append([k,'l1_distance' if p == 1 else 'l2_distance',accuracy])
# y_pred = knn.predict(x_test)
#     求出预测准确率
# accuracy = accuracy_score(y_test,y_pred) * 100
# print("预测的准确率：", accuracy)
df = pd.DataFrame(result_list, columns=['k','距离函数', '预测准确率'])
print(df)

# print( np.abs(x_train - x_test[0].reshape(1, -1)))

