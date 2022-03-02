from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# 加载数据集并进行归一化预处理
def loadDataSet():
    boston_dataset = load_boston()
    X = boston_dataset.data
    y = boston_dataset.target
    y = y.reshape(-1, 1)
    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 分别初始化对特征和目标值的标准化器
    ss_X, ss_y = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

    # 分别对训练和测试数据的特征以及目标值进行标准化处理
    X_train, y_train = ss_X.fit_transform(X_train), ss_y.fit_transform(y_train)
    X_test, y_test = ss_X.transform(X_test), ss_y.transform(y_test)
    y_train, y_test = y_train.reshape(-1, ), y_test.reshape(-1, )
    return X_train, X_test, y_train, y_test


def trainRF(X_train, y_train):
    model_rf = RandomForestRegressor(n_estimators=10000)
    model_rf.fit(X_train, y_train)
    return model_rf


def test(model, X_test, y_test):
    y_pre = model.predict(X_test)
    print("The mean root mean square error of RF-Model is {}".format(mean_squared_error(y_test, y_pre)**0.5))
    print("The mean squared error of RF-Model is {}".format(mean_squared_error(y_test, y_pre)))
    print("The mean absolute error of RF-Model is {}".format(mean_absolute_error(y_test, y_pre)))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadDataSet()
    # 训练RF模型
    model = trainRF(X_train, y_train)
    test(model, X_test, y_test)