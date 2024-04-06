import torch.nn.functional as F
import math
import torch
import os
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np


class PositionEncoder(nn.Module):
    def __init__(self, embedding_num, max_len, device, dropout=0.3):  # embedding=307
        # 先变成308，最后在变成307
        super().__init__()
        self.embedding_num = embedding_num  # 307
        self.max_len = max_len  # max_len=32
        self.pe = torch.zeros(self.max_len, self.embedding_num, device=device)  # torch.Size([32, 308])

        position = torch.arange(1, self.max_len+1, dtype=torch.float).unsqueeze(1) # torch.Size([32, 1])
        # div_term.shape=torch.Size([154])
        div_term = torch.exp(torch.arange(0, self.embedding_num, 2).float() * (-math.log(10000.0) / self.embedding_num))
        # div_term = torch.rand(size=(50,)).sort(descending=True)[0]
        self.pe[:, 0::2] = torch.sin(position * div_term)  # (position * div_term).shape=torch.Size([32, 154]),self.pe[:, 0::2].shape=torch.Size([32, 154])
        self.pe[:, 1::2] = torch.cos(position * div_term)  # self.pe[:, 1::2].shape=torch.Size([32, 153])

        self.dropout = nn.Dropout(dropout)
        # self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        # 将308变成307
        # self.pe=self.pe[:,:307]

    def forward(self, x):  # x.shape=torch.Size([8, 32, 307])
        x = x + self.pe[:x.shape[1]]
        x = self.dropout(x)
        return x


# 单层（attention，feedforward ）
class EncoderBlock(nn.Module):
    def __init__(self, embedding_num, n_heads, ff_num):
        super().__init__()
        self.embedding_num = embedding_num
        self.n_heads = n_heads

        # --------------------------  MultiHeadAttention -------------------
        self.W_Q = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.W_K = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.W_V = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.fc = nn.Linear(self.embedding_num, self.embedding_num, bias=False)
        self.att_ln = nn.LayerNorm(self.embedding_num)

        # -------------------------- 定义卷积层 ------------------------------
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=(1, 1))  
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        # --------------------------- FeedForward  --------------------------
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_num, ff_num, bias=False),
            nn.ReLU(),
            nn.Linear(ff_num, self.embedding_num, bias=False)
        )
        self.feed_ln = nn.LayerNorm(self.embedding_num)

    # def forward(self, x, attn_mask):
    def forward(self, x, max_len, embedding_num):
        # ---------------------- MultiHeadAttention forward ------------------
        Q = self.W_Q(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2) # x.shape=torch.Size([8, 32, 307])=[batch_size,max_len,embedding_num]
        K = self.W_K(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
        V = self.W_V(x).reshape(*x.shape[:2], self.n_heads, -1).transpose(1, 2)
        # attn_mask_new = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.embedding_num / self.n_heads)
        # scores.masked_fill_(attn_mask_new.type(torch.bool), -1e9)

        attn = F.softmax(scores, dim=-1)
        context = (attn @ V).transpose(1, 2).reshape(*x.shape)

        att_result = self.fc(context)
        att_result = self.att_ln(x + att_result)  # att_result.shape=torch.Size([8, 32, 306])=[batch_size,max_len,embedding_num]

        # ---------------------- 卷积 forward ------------------
        # 调整输入的形状以符合Conv2d的期望格式 [batch_size, channels, height, width]
        x = x.view(-1, 1, max_len, embedding_num)
        # x = x.relu(self.conv1(x))
        x = self.conv1(x)
        x = self.pool1(x)
        # x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # 展平，准备连接到全连接层
        x = x.view(x.size(0), -1)
        # x = self.conv_fc(x)
        # 重构输出形状以符合期望的输出 [batch_size, seq_length, features]
        conv_output = x.view(-1, max_len, embedding_num)  # conv_output.shape=torch.Size([8, 32, 306])=[batch_size,max_len,embedding_num]

        # ---------------------- concat-修改shape ------------------
        output = torch.cat((att_result, conv_output), dim=1)  # torch.Size([8, 64, 306])=[batch_size,2*max_len,embedding_num]

        output = output.permute(0,2,1)
        output = F.avg_pool1d(output,kernel_size=2,stride=2)
        att_result = output.permute(0,2,1)  # 和cnn合并后的att_result

        #  ----------------------- FeedForward forward --------------------
        feed_result = self.feed_forward(att_result)
        feed_result = self.feed_ln(att_result + feed_result)

        return feed_result, attn
        


class TransformerEncoder(nn.Module):
    def __init__(self,device,block_nums=2, embedding_num=200, max_len=100, n_heads=3, ff_num=128):  # block_nums代表堆叠多少层，ff_num前馈神经网络的隐藏层的维度
        super().__init__()
        self.device = device
        # ----------------------- position encoder --------------------------
        self.position_encoder = PositionEncoder(embedding_num, max_len, device)

        # ---------------------------- blocks -------------------------
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embedding_num, n_heads, ff_num) for _ in range(block_nums)])

        # ------------------------ classifiy and Loss------------------------
        # self.classifier = nn.Linear(embedding_num, params["class_num"])
        # self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, batch_x, max_len, embedding_num):  # batch_x就是每个批次的数据
        # ------------------- position embedding -----------------
        encoder_output = self.position_encoder.forward(batch_x)

        # -------------------- blocks forward --------------------
        # enc_self_attn_mask = get_attn_pad_mask(datas_len, self.device)
        for block in self.encoder_blocks:
            encoder_output, _ = block.forward(encoder_output, max_len, embedding_num)

        return encoder_output

        
