import numpy as np
from matplotlib import pyplot as plt
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

from mod.seq2seq import seq2seq


def sinx_plot():
    x = np.linspace(-1, 1, 100)  # 1x100维
    y = np.sin(10 * x)  # 1x100
    x_1 = x.copy()
    plt.plot(x, y)
    x = torch.tensor(x, dtype=torch.float).view(100, 1, 1)
    y = torch.tensor(y, dtype=torch.float)

    net = seq2seq()

    opt = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss().cuda()

    for i in range(200):
        opt.zero_grad()
        y_ = net(x[:, :, :])
        loss = loss_func(y, y_)
        loss.backward()
        print(f"epoch{i}损失", loss)
        opt.step()

    y_ = net(x)

    plt.plot(x_1, y_.reshape(-1).detach().numpy())
    plt.plot(x_1[:40], y[:40])

    plt.legend(['sin函数计算','神经网络预测','一半的x预测完整的y'])
    plt.xlabel("x")
    plt.ylabel("x")

    plt.show()


if __name__ == "__main__":
    sinx_plot()
