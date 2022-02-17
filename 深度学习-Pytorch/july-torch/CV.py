import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

print(torch.__version__)

class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))

        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self,x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x



# mnist_data
batch_size = 32

lr = 1e-4
momentum = 0.5

train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("./mnist_data",train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.1306,0.308)
                            ])),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)

test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("./mnist_data",train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.1306,0.308)
                            ])),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
)

def train(model, device, train_dataloader, optimizer, epoch):
    model.train()
    for idx, (data,target) in enumerate(train_dataloader):
        # print(data.shape,target.shape)
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.nll_loss(pred, target)
        # break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Train Epoch:{}, iteration:{}, Loss: {}".format(epoch,idx,loss.item()))


def test(model, device, test_dataloader, optimizer, epoch):
    model.eval()
    total_loss = 0
    correct = 0.

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_dataloader):
            # print(data.shape,target.shape)
            data, target = data.to(device), target.to(device)
            pred = model(data)
            total_loss += F.nll_loss(pred, target, reduction="sum")
            out = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(out)).sum().items()

        total_loss /= len(test_dataloader)

        print("Test loss: {}, accuracy:{}".format(total_loss, correct))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()
optimizer =  torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 1
for epoch in range(1):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader, optimizer, epoch)

# torch.save(model.state_dict(), 'mnist_cnn.pt')


