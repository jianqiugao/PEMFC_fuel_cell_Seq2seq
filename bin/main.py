import numpy as np

from lib.三角函数拟合 import sinx_plot, plt

# sinx_plot()

import torch
import math

# y = torch.exp(torch.arange(0,500,2)*-math.log(10000)/500).detach().numpy()
# plt.plot(range(len(y)),y)
# plt.show()
# print(y.shape)

# data = np.triu(np.ones((5,5)),k=0)
# plt.imshow(data)
# plt.show()
# print(data)

# x =[1,2,3,4]
# y =[1,2,3]
# data = [[m,n] for m,n in zip(x,y)]
# print(data)

# x = torch.arange(0, 24., 1).view(2, 1, 3, 4)
# print(x)
# x_1 = x.mean(-1, keepdim=True)
#
# print(x*(x-x_1))
# y = x.transpose(-1, -2)
# print(x.shape)
# print(y.shape)
# y = torch.matmul(x, y)
# print(torch.sum(y.reshape(-1)))
# print(y.shape)

list = torch.Tensor([[1,2,3,4,5],[4,5,6,7,8],[8,9,10,11,12]])
np.triu(np.ones((3,3)),k=-1)
data = torch.Tensor(
    [[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])
print(data)
data = torch.matmul(data,list)
print(data)
transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
