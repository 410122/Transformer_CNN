# Pems数据集
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from Transformer_Encoder import TransformerEncoder



# 创建数据集dataset
class my_dataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
        # 计算每个窗口的步长为32，时间窗口的总数
        self.window_size = 32
        self.number_of_windows = data_tensor.shape[0] // self.window_size

    def __getitem__(self,index):
        # 计算开始和结束的索引
        start_index = index * self.window_size
        end_index = start_index + self.window_size
        # 返回对应的时间窗口
        return self.data_tensor[start_index:end_index, :]

    def __len__(self):
        # 返回时间窗口的总数
        return self.number_of_windows

class TransformerClassifier(nn.Module):
    def __init__(self,embedding_num,max_len):
        super().__init__()
        self.transformer=TransformerEncoder(device=device,block_nums=2, embedding_num=embedding_num, max_len=max_len, n_heads=2, ff_num=128)

    def forward(self,batch_embedding):
        out=self.transformer(batch_embedding, max_len, embedding_num)  # # out输出的形状应该和batch_embedding的形状相同
        return out

if __name__=="__main__":
    # 读取数据
    pems04_data = np.load("D:\\00_New_Inbox\\trans_cnn\\data\\PEMS04\\pems04.npz")
    data = pems04_data['data']
    # data_tensor = torch.tensor(data)  # 将NumPy数组转换为PyTorch张量
    data_tensor = torch.FloatTensor(data) # 将NumPy数组转换为PyTorch张量(不能用上一行的操作)
    data_tensor = data_tensor[:, :, 0]  # 只保留交通流量特征 (假设它是在最后一个维度的第一个位置)
    # print(data_tensor.shape)  # 输出形状应该是 data_tensor：torch.Size([16992, 307])

    data_tensor=data_tensor[:,:-1]  # 307改成306  torch.Size([16992, 306])
    print(data_tensor.shape)  # torch.Size([16992, 306])

    # 数据归一化
    

    # 定义常量
    batch_size=8
    max_len=32
    embedding_num=306
    epoch=2

    # gpu运行
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 创建dataset与Dataloader
    train_dataset=my_dataset(data_tensor)
    train_dataloader=DataLoader(train_dataset,batch_size)

    # 实例化模型
    model = TransformerClassifier(embedding_num, max_len).to(device)

    for e in range(epoch):
        for batch_data in train_dataloader:
            batch_data=batch_data.to(device)
            # final_out=model(batch_data,max_len,embedding_num)  
            final_out=model(batch_data)   # batch_data.device=cpu

            print(final_out)
            print(f'输入特征尺寸:{batch_data.shape}，输出特征尺寸:{final_out.shape}')

    print('finish!')