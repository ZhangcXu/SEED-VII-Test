import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MultiViewEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, heads):
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads

        self.transform1 = nn.Linear(input_dim, output_dim)
        
         # 为每个头创建独立的卷积序列
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        kernel_size = 5
        conv_channels = 5
        pool_size = 2
        for _ in range(heads):
            # 卷积层 + ReLU + 最大池化
            conv_seq = nn.Sequential(
                nn.Conv1d(
                    in_channels=1, 
                    out_channels=conv_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2  # 保持序列长度不变
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size)
            )
            self.conv_layers.append(conv_seq)
            
            # 计算卷积池化后的序列长度
            conv_output_len = (input_dim + 2*(kernel_size//2) - kernel_size + 1)
            pool_output_len = (conv_output_len - pool_size) // pool_size + 1
            
            # 线性层将特征映射到输出维度
            linear = nn.Linear(conv_channels * pool_output_len, output_dim)
            self.linear_layers.append(linear)
            
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        B, _ = x.size()
        x1 = self.transform1(x).unsqueeze(1).repeat(1, self.heads, 1)
        
        x = x.unsqueeze(1)  
        head_outputs = []
        for i in range(self.heads):
            # 卷积部分
            conv_out = self.conv_layers[i](x)  # (batch, conv_channels, pooled_len)
            
            # 展平特征
            flattened = conv_out.view(B, -1)  # (batch, conv_channels * pooled_len)
            
            # 线性层
            out_head = self.linear_layers[i](flattened)  # (batch, output_dim)
            head_outputs.append(out_head)
        
        # 组合所有头输出
        x2 = torch.stack(head_outputs, dim=1) 

        x = torch.mul(x1, x2)
        x = self.bn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x