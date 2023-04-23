import numpy as np
import pandas as pd
from config import config
from 三角函数拟合 import plt
from mod.工具 import normal_data

data = pd.read_csv('../data/fuelcell.csv')[1:]
item = data.columns

for i in item:
    data[i] = data[i].map(lambda x: normal_data(x,config[i][1],config[i][0]))


# fig,axes = plt.subplots(5,1)
# axes[0].plot(data.index,data['电流'])
# axes[1].plot(data.index,data['电压'])
# axes[2].plot(data.index,data['电功'])
# axes[3].plot(data.index,data['热功'])
# axes[4].plot(data.index,data['温度'])
# plt.show()

# 把拉载的电流当做唯一的操作变量，由前10个时刻预测后面10个时刻
# 前10个时刻的电流、电压、电功、热功、温度 5个信息输入编码器
# 解码器拿到编码器的contex，然后添加要预测时刻的拉载电流
data_y = data.shift(-10).dropna()

train_x = data.values[:-11, :].reshape(-1, 10, 5)  # 训练的x
train_y = data_y.values[:-1, :].reshape(-1, 10, 5)[:, :, 1:5]  # 训练的y
manual = data_y.values[:-1, :].reshape(-1, 10, 5)[:, :, 0].reshape(-1, 10, 1)  # 操作
print(train_y.shape)  # batch ,time, content
print(train_x.shape)  # batch ,time, content
print(manual.shape)  # batch ,time, content

np.save('../data/train_x', train_x)
np.save('../data/train_y', train_y)
np.save('../data/manual', manual)

# 训练样本偏少了只有1492个，每个只有10分钟
