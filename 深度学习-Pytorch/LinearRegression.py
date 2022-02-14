import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# 数据生成
car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np = car_price_np.reshape(-1,1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))

# lets define number of car sell
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

# lets visualize our data
# import matplotlib.pyplot as plt
# plt.scatter(car_prices_array,number_of_car_sell_array)
# plt.xlabel("Car Price $")
# plt.ylabel("Number of Car Sell")
# plt.title("Car Price$ VS Number of Car Sell")
# plt.show()



# 模型构建
class LinearRegression(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim)

mse = nn.MSELoss()
learning_rate = 0.02
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_list = []
iteration_number = 1001

# 训练
for iteration in range(iteration_number):
    optimizer.zero_grad()

    results = model(car_price_tensor)
    loss = mse(results, number_of_car_sell_tensor)

    loss.backward()

    optimizer.step()

    loss_list.append(loss.data)

    if(iteration%50==0):
        print("epoch {},loss {}".format(iteration,loss.data))

print(loss_list)

plt.plot(range(iteration_number),loss_list)
plt.xlabel("iteration number")
plt.ylabel("Loss")
plt.show()


