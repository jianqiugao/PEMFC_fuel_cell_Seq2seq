import torch
from torch import nn


class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()
        # self.embedding = nn.Embedding(1, 10)
        self.encoder = nn.GRU(1, 30)
        self.decoder = nn.GRU(30, 30)
        self.linear = nn.Linear(30, 1)

    def forward(self, x):
        size = 100 #x.shape[0]
        # x = self.embedding(x)
        res = torch.ones(size)
        contex, hidden = self.encoder(x)

        input = torch.ones_like(hidden)  # 让他知道开始输出
        for i in range(size):  #
            hidden, input = self.decoder(hidden, input)
            res_ = self.linear(input)
            res[i] = res_
        return res
