import torch.nn as nn
import torch
from torch.autograd import Variable
class ResidualBlock(nn.Module):
    def __init__(self, channel_size, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )
    def forward(self, x):
        x = self.block(x)
        return x
class Shallow_Feature_Extraction_Sub_network(nn.Module):
    def __init__(self, in_channels, num_res_layers, kernel_size, channel_size):
        super(Shallow_Feature_Extraction_Sub_network, self).__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())
        res_layers = [ResidualBlock(channel_size, kernel_size) for _ in range(num_res_layers)]
        self.res_layers = nn.Sequential(*res_layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=32, kernel_size=kernel_size, padding=padding)
        )
    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x
