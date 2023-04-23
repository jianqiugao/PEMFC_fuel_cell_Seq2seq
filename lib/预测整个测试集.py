import torch
import numpy as np
import pandas as pd

from mod.遗传算法 import GA
from mod.seq2seq_fuelcell import seq2seq_fuelcell
from mod.工具 import normal_data, unormal_data
from config import config
from 三角函数拟合 import plt

net = seq2seq_fuelcell()
net.load_state_dict(torch.load('mode.pt'))

train_x = torch.Tensor(np.load('../data/train_x.npy'))
train_y = torch.Tensor(np.load('../data/train_y.npy'))
manual = torch.Tensor(np.load('../data/manual.npy'))

train_y = torch.cat((train_y, manual), dim=2)  # 粘贴回去放到datset

data_test = train_x[:, :, :]
manual_y = train_y[:, :, -1].unsqueeze(-1)
real_data = train_y[:, :, 0:4].reshape(-1,4).detach().numpy()
net.eval()
pre_data = net(data_test,manual_y).reshape(-1,4).detach().numpy()


fig,axe = plt.subplots(2,2)
len = pre_data[:,0].shape[0]
axe[0,0].set_title('电压')
axe[0,0].plot(range(len),pre_data[:,0])
axe[0,0].plot(range(len),real_data[:,0])
axe[0,0].legend(['预测值','真实值'])
axe[0,0].set_ylim([0,1])

axe[0,1].set_title('电功')
axe[0,1].plot(range(len),pre_data[:,1])
axe[0,1].plot(range(len),real_data[:,1])
axe[0,1].legend(['预测值','真实值'])
axe[0,1].set_ylim([0,1])

axe[1,0].set_title('热功')
axe[1,0].plot(range(len),pre_data[:,2])
axe[1,0].plot(range(len),real_data[:,2])
axe[1,0].legend(['预测值','真实值'])
axe[1,0].set_ylim([0,1])

axe[1,1].set_title('温度')
axe[1,1].plot(range(len),pre_data[:,3])
axe[1,1].plot(range(len),real_data[:,3])
axe[1,1].legend(['预测值','真实值'])
axe[1,1].set_ylim([0,1])

plt.show()

print('hello')

