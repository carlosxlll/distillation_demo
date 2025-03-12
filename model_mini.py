import torch
import torch.nn as nn
class Encoder_mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),  # 增大通道数
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),  # 对称结构
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)