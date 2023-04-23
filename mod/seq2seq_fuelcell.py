import torch
from torch import nn


class seq2seq_fuelcell(nn.Module):
    def __init__(self):
        super(seq2seq_fuelcell, self).__init__()
        self.encoder = nn.GRU(5, 128)
        self.decoder = nn.GRU(129, 256)  # 64+未来的操作这一个维度哈
        self.linear = nn.Linear(256, 4)

    def forward(self, x, y):
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        seq,batch,content = x.shape
        content_ = 4
        size = 10  # 初始化用来表示预测的长度
        res = torch.ones(seq,batch,content_)


        out, contex = self.encoder(x)  # 编码器进行编码[seq,batch,content]
        cat_x = y[0, :, :].unsqueeze(0)
        contex_ = torch.cat((contex, cat_x), dim=2)
        _, shape = self.decoder(contex_)
        input = torch.ones_like(shape)  # 让他知道开始输出,也就是初始化一个输出
        for i in range(size):  #
            cat_x = y[i, :, :].unsqueeze(0)
            contex_ = torch.cat((contex, cat_x), dim=2)
            hidden, input = self.decoder(contex_, input)
            res_ = self.linear(input)
            res_ = res_
            res[i] = res_
        return res.permute(1,0,2)


if __name__ == "__main__":
    net = seq2seq_fuelcell()
    x = torch.ones(10,2,5)
    y = torch.ones(10,2,1)
    print(net(x, y)) # 加入操作预测y
