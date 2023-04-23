from mod.seq2seq_fuelcell import seq2seq_fuelcell
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.三角函数拟合 import plt

train_x = torch.Tensor(np.load('../data/train_x.npy'))
train_y = torch.Tensor(np.load('../data/train_y.npy'))
manual = torch.Tensor(np.load('../data/manual.npy'))

train_y = torch.cat((train_y, manual), dim=2)  # 粘贴回去放到datset
data_train = TensorDataset(train_x[:1000, :, :], train_y[:1000, :, :])
data_test = TensorDataset(train_x[1000:, :, :], train_y[1000:, :, :])

net = seq2seq_fuelcell()
opt = torch.optim.Adam(net.parameters())
loss_func = torch.nn.L1Loss()

dataset = DataLoader(
    dataset=data_train,  # torch TensorDataset format
    batch_size=180,  # mini batch size
    shuffle=False,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
)

trian_loss = []
test_loss = []
for epoch in range(300):
    l_ = []
    for x, y in dataset:
        opt.zero_grad()
        y_1 = y[:, :, -1].unsqueeze(-1)
        y_pred = net(x, y_1)
        l = loss_func(y_pred, y[:, :, 0:4])
        l_.append(l)
        l.backward()
        opt.step()
    y_test = net(train_x[1000:, :, :], train_y[1000:, :, :][:, :, -1].unsqueeze(-1))
    l__ = loss_func(y_test, train_y[1000:, :, :][:, :, 0:4])
    l_ = sum(l_)
    trian_loss.append(l_.detach().numpy())
    test_loss.append(l__.detach().numpy())
    print(f"第：{epoch}轮loss:{l_}")
torch.save(net.state_dict(),'mode.pt')
plt.plot(range(len(trian_loss)), trian_loss)
plt.plot(range(len(test_loss)), test_loss)
plt.legend(['训练集','测试集'])
plt.show()
