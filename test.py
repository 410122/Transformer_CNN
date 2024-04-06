import torch
import torch.nn as nn
import math

class PositionEncoder(nn.Module):
    def __init__(self, embedding_num, max_len, device, dropout=0.3):
        super().__init__()
        self.embedding_num = embedding_num
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, self.embedding_num, device=device)

        position = torch.arange(1, self.max_len+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_num, 2).float() * (-math.log(10000.0) / self.embedding_num))
        # div_term = torch.rand(size=(50,)).sort(descending=True)[0]
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        # self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):  # x.shape=torch.Size([8, 32, 307])
        x = x + self.pe[:x.shape[1]]
        x = self.dropout(x)
        return x

if  __name__=="__main__":
    x = torch.rand(8, 32, 307)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=x.to(device)
    model=PositionEncoder()