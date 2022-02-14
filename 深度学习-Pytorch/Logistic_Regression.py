# Import Libraries
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Prepare Dataset
# load data
train = pd.read_csv(r"mnist/train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42)

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()


# class LogisticRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegressionModel, self).__init__()
#         # Linear part
#         self.linear = nn.Linear(input_dim, output_dim)
#         # There should be logistic function right?
#         # However logistic function in pytorch is in loss function
#         # So actually we do not forget to put it, it is only at next parts
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel,self).__init__()

        # self.linear = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(150, 150)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(150, output_dim)

    def forward(self,x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

input_dim = 28*28
output_dim = 10

model = LogisticRegressionModel(input_dim,output_dim)
error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(20000):
    for i, (images,labels) in enumerate(train_loader):

        train = Variable(images.view(-1,input_dim))
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(train)

        loss = error(outputs, labels)
        loss.backward()

        optimizer.step()

        count += 1

        if count % 50 == 0:
            total = 0
            correct = 0

            for images, labels in test_loader:
                test = Variable(images.view(-1,input_dim))

                outputs = model(test)
                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)

                correct += (predicted ==  labels).sum()

            accuracy = 100 * correct / float(total)
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if count % 500 == 0:
            print("Iteration: {} Loss: {} Accuracy: {} %".format(count,loss.data, accuracy ) )





























