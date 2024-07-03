"""
Activations
"""

import math
import torch
import torch.nn as nn


class SequecialHGELUV4B(nn.Module):
    """
    串行实现版本，不包含 ReLU层
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            r: int = 16,
            dropout_p: float = 0,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_features, num_features//r)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0. else nn.Identity()
        self.fc21 = nn.Linear(num_features//r, num_features)
        self.fc22 = nn.Linear(num_features//r, num_features)
        self.eps = eps

    def encode(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc21(x), self.fc22(x)

    def forward(self, x):
        mu, log_var = self.encode(torch.flatten(self.avg_pool(x), 1))
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 归一化
        x_dim = x.ndim
        if x_dim == 2:
            norm_out = (x - mu) / (std + self.eps)
        elif x_dim == 4:
            b, c, _, _ = x.size()
            norm_out = (x - mu.reshape(b, c, 1, 1)) / (std.reshape(b, c, 1, 1) + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x
