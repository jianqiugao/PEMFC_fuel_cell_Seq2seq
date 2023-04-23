import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# embdding = nn.Embedding(10, 3, padding_idx=0)  # 后面参数可以确定把哪一个变成0
# input1 = torch.LongTensor([[1, 2, 0, 4], [4, 3, 2, 9]])
# print(embdding(input1))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model 词嵌入的维度
        # vocab 词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 521, 509], [491, 998, 1, 221]]))
embr = Embeddings(d_model, vocab)
embr_results = embr(x)
print(embr_results.shape)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model 词嵌入的维度
        # dropout
        # max_len
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 初始化编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义一个变换矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


dropout = 0.1

x = embr_results
pe = PositionalEncoding(d_model, dropout, max_len=60)
pe_results = pe(x)


# print(pe_results)

# 构建掩码张量的函数
def subsequent_mask(size):
    # size : 掩码张量最后两个维度，形成一个方阵
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 使得三角矩阵进行反转
    return torch.from_numpy(1 - subsequent_mask)


size = 4
sm = subsequent_mask(size)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, 0)

    p_attn = F.softmax(score, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_results
attn, p_attn = attention(query, key, value, mask=sm)
print(attn.shape)
