import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim  # 引入PyTorch自带的可选优化函数






net = models.AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optimizer.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(30):

    for i, data in enumerate(traindata):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()