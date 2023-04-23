import torch
import numpy as np
import pandas as pd

from mod.遗传算法 import GA
from mod.seq2seq_fuelcell import seq2seq_fuelcell
from mod.工具 import normal_data, unormal_data
from config import config
from 三角函数拟合 import plt
import copy

# 设定一个损失，做一个代价函数，让这个代价函数最小
net = seq2seq_fuelcell()
net.load_state_dict(torch.load('mode.pt'))
net.eval()
state = np.array(
    [[[0.132378208, 498.3820802, 0.065974927, 0.000374929, 20.35818197],
      [0.170700803, 496.5324424, 0.084758487, 0.000799201, 20.35827468],
      [0.222321497, 494.606599, 0.10996168, 0.001469043, 20.35839953],
      [0.305417228, 492.2853429, 0.150352425, 0.002727039, 20.35860091],
      [0.418370848, 489.9750039, 0.204991258, 0.004702159, 20.35887627],
      [0.557452854, 487.85643, 0.271956959, 0.007446353, 20.35921896],
      [0.738578799, 485.765195, 0.358775874, 0.011410364, 20.3596729],
      [0.991142597, 483.5584959, 0.479275423, 0.017499744, 20.36032302],
      [1.336658825, 481.2858846, 0.643315025, 0.026637729, 20.36124945],
      [1.791785325, 479.0208057, 0.85830245, 0.039766603, 20.36254287]]])
# 电流，电压，电功，热功，温度

# 找一组最优的操作电流
# 输出

输出 = np.array([[[476.7365939, 1.14047891, 0.058557497, 20.3643895],
                  [474.9266861, 1.420436188, 0.078622727, 20.36635863],
                  [473.5833126, 1.665438311, 0.097169566, 20.36813092],
                  [472.6332138, 1.859659204, 0.112457777, 20.36963517],
                  [471.7622138, 2.053880102, 0.128224083, 20.37123137],
                  [470.6903144, 2.315130371, 0.150135354, 20.37352882],
                  [468.9767178, 2.786161385, 0.191522108, 20.37815339],
                  [467.0790091, 3.387667557, 0.247579923, 20.38498777],
                  [465.3116861, 4.022686226, 0.31038418, 20.3934078],
                  [463.585325, 4.700173404, 0.381511688, 20.40383915],
                  ]])


# 电压，电压，电功，热功，温度

# 根据自己的任务来做

# 温度接近60度，热功0.2kw，电功尽可能高，电压不超过450v

# 代价函数  =
def normal_input(data):
    data = data.squeeze(0)
    data[:, 0] = np.array([normal_data(x, config['电流'][1], config['电流'][0]) for x in data[:, 0]])
    data[:, 1] = np.array([normal_data(x, config['电压'][1], config['电压'][0]) for x in data[:, 1]])
    data[:, 2] = np.array([normal_data(x, config['电功'][1], config['电功'][0]) for x in data[:, 2]])
    data[:, 3] = np.array([normal_data(x, config['热功'][1], config['热功'][0]) for x in data[:, 3]])
    data[:, 4] = np.array([normal_data(x, config['温度'][1], config['温度'][0]) for x in data[:, 4]])
    return data


def unormal_output(data):
    data[:, 0] = np.array([unormal_data(x, config['电压'][1], config['电压'][0]) for x in data[:, 0]])
    data[:, 1] = np.array([unormal_data(x, config['电功'][1], config['电功'][0]) for x in data[:, 1]])
    data[:, 2] = np.array([unormal_data(x, config['热功'][1], config['热功'][0]) for x in data[:, 2]])
    data[:, 3] = np.array([unormal_data(x, config['温度'][1], config['温度'][0]) for x in data[:, 3]])
    return copy.deepcopy(data)


state = normal_input(state)


def fun(x):
    x = torch.Tensor(x).reshape(-1, 10, 1)
    ops = x.shape[0]
    前十秒的状态 = torch.Tensor(state).reshape(-1, 10, 5).tile(ops, 1, 1)
    output = net(前十秒的状态, x).detach().numpy()
    output = unormal_output(output)
    res = (abs(np.mean(output[:, :, 3], axis=1) - 45)) + abs(np.mean(output[:, :, 0], axis=1) - 460)
    # print(res.max(), res.min())
    return res


lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ub = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ga = GA(func=lambda x: fun(x),
        n_dim=10, size_pop=60, max_iter=220, lb=lb, ub=ub,
        precision=[0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001])
# # 其中lb和up代表的是取值范围
best_x, best_y = ga.run()

best_x = torch.Tensor(best_x).reshape(-1, 10, 1)

initial = np.array(
    [[[0.132378208, 498.3820802, 0.065974927, 0.000374929, 20.35818197],
      [0.170700803, 496.5324424, 0.084758487, 0.000799201, 20.35827468],
      [0.222321497, 494.606599, 0.10996168, 0.001469043, 20.35839953],
      [0.305417228, 492.2853429, 0.150352425, 0.002727039, 20.35860091],
      [0.418370848, 489.9750039, 0.204991258, 0.004702159, 20.35887627],
      [0.557452854, 487.85643, 0.271956959, 0.007446353, 20.35921896],
      [0.738578799, 485.765195, 0.358775874, 0.011410364, 20.3596729],
      [0.991142597, 483.5584959, 0.479275423, 0.017499744, 20.36032302],
      [1.336658825, 481.2858846, 0.643315025, 0.026637729, 20.36124945],
      [1.791785325, 479.0208057, 0.85830245, 0.039766603, 20.36254287]]])


前十秒的状态 = torch.Tensor(state).reshape(-1, 10, 5)

output = net(前十秒的状态, best_x)

pre_data = output.detach().numpy() # 优化后的数据组转换为2维数组

pre_data = unormal_output(pre_data)

# 代价函数值优化后 = (abs(np.mean(pre_data[:, 3]) - 60)) + abs(np.mean(pre_data[:, 0]) - 460)
代价函数值优化后 = (abs(np.mean(pre_data[:, :, 3], axis=1) - 45)) + abs(np.mean(pre_data[:, :, 0], axis=1) - 460)


raw_man = [2.392262151, 2.990853599, 3.516674398, 3.934677355, 4.353634186, 4.918585108, 5.940937534, 7.252879044,
          8.645143344, 10.13874502]

raw_man = torch.Tensor([normal_data(x, config['电流'][1], config['电流'][0]) for x in raw_man]).reshape(1, 10, 1)


real_data = net(前十秒的状态, raw_man).detach().numpy()  # 优化前
real_data=unormal_output(real_data)
# 代价函数值优化前 = (abs(np.mean(real_data[:, 3]) - 60)) + abs(np.mean(real_data[:, 0]) - 460)
代价函数值优化前=(abs(np.mean(real_data[:, :, 3], axis=1) - 45)) + abs(np.mean(real_data[:, :, 0], axis=1) - 460)

print('代价函数值优化前', 代价函数值优化前)
print('代价函数值优化后', 代价函数值优化后)
# print(best_y)


real_data = real_data.squeeze(0)
pre_data = pre_data.squeeze(0)
fig, axe = plt.subplots(2, 2)
len = pre_data[:, 0].shape[0]
axe[0, 0].set_title('电压')
axe[0, 0].plot(range(len), pre_data[:, 0])
axe[0, 0].plot(range(len), real_data[:, 0])
axe[0, 0].legend(['优化后', '优化前'])
# axe[0, 0].set_ylim([0, 1])

axe[0, 1].set_title('电功')
axe[0, 1].plot(range(len), pre_data[:, 1])
axe[0, 1].plot(range(len), real_data[:, 1])
axe[0, 1].legend(['优化后', '优化器'])
# axe[0, 1].set_ylim([0, 1])

axe[1, 0].set_title('热功')
axe[1, 0].plot(range(len), pre_data[:, 2])
axe[1, 0].plot(range(len), real_data[:, 2])
axe[1, 0].legend(['优化后', '优化器'])
# axe[1, 0].set_ylim([0, 1])

axe[1, 1].set_title('温度')
axe[1, 1].plot(range(len), pre_data[:, 3])
axe[1, 1].plot(range(len), real_data[:, 3])
axe[1, 1].legend(['优化后', '优化器'])
# axe[1, 1].set_ylim([0, 1])
plt.show()
#
best_x = [unormal_data(x.detach().numpy(), config['电流'][1], config['电流'][0]) for x in best_x.reshape(-1)]
raw_man = [2.392262151, 2.990853599, 3.516674398, 3.934677355, 4.353634186, 4.918585108, 5.940937534, 7.252879044,
          8.645143344, 10.13874502]
print('原始操作',raw_man)
print('优化后的操作',best_x)
plt.close()
plt.plot(range(10),raw_man)
plt.plot(range(10),best_x)
plt.legend(['原始操作','优化操作'])
plt.xlabel('时间步')
plt.ylabel('电流')
plt.show()


