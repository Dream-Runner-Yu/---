import torch

a = torch.empty(5,3)

x = torch.rand(5,3)

zero = torch.zeros(5,3,dtype=torch.long)

# x.types

x = torch.tensor([5,5,3])

x = x.new_ones(5,3,dtype=torch.double)

torch.rand_like(x,dtype=torch.float)

y = torch.rand(5,3)

x + y

torch.add(x,y)

out = torch.empty(5,3)

# y += x
y.add_(x)
