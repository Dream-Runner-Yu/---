from sklearn import datasets
from matplotlib import pyplot as plt

boston_dataset = datasets.load_boston()
print(boston_dataset['DESCR'])

x_data = boston_dataset.data
y_data = boston_dataset.target
feature_data = boston_dataset.feature_names

for i in range(13):
    plt.subplot(5, 3, i + 1)
    plt.scatter(x_data[:, i], y_data, s=20)
    plt.title(feature_data[i])

plt.show()