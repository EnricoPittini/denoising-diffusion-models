import math
from inspect import isfunction
from functools import partial
from einops import rearrange
from utils.encodings import pos_encoding

import torch
import torch.nn.functional as F
from torch import nn, einsum


class Block(nn.Module):
    """Block for grouped convolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    groups : int, optional
        default is 1
    device : torch.device, optional
    """
    def __init__(self, in_channels, out_channels, groups=1, device='cpu'):
        super().__init__()

        self.resize = in_channels != out_channels

        middle_channels = min(in_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(num_features=in_channels, device=device)
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, padding=0, device=device)
        self.bn2 = nn.BatchNorm2d(num_features=middle_channels, device=device)
        self.c2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, padding=1, groups=groups, device=device)
        self.bn3 = nn.BatchNorm2d(num_features=middle_channels, device=device)
        self.c3 = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, padding=0, device=device)

        if self.resize:
            self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device)

    def forward(self, x):
        xr = self.bn1(x)
        xr = F.relu(xr)
        xr = self.c1(xr)

        xr = self.bn2(xr)
        xr = F.relu(xr)
        xr = self.c2(xr)

        xr = self.bn3(xr)
        xr = F.relu(xr)

        if self.resize:
            x = self.res_conv(x)

        x = self.c3(xr) + x

        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, compression=1, device='cpu'):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(in_channels=channels, out_channels=channels//compression, kernel_size=1, device=device)
        self.fc2 = nn.Conv2d(in_channels=channels//compression, out_channels=channels, kernel_size=1, device=device)

    def forward(self, x):
        xr = self.squeeze(x)

        xr = self.fc1(xr)
        xr = F.relu(xr)

        xr = self.fc2(xr)
        xr = F.sigmoid(xr)

        return x*xr


class FifthModel(nn.Module):
    """Uses X blocks with Gruped Convolution and Squeeze-and-Excitation."""
    def __init__(self, in_channels=3, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=in_channels+temp_encoding_initial_channels, out_channels=block_channels[0], kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            block = Block(in_channels=block_channels[i], out_channels=block_channels[i+1], device=device)
            squeeze_and_excitation = SqueezeAndExcitation(channels=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((block, squeeze_and_excitation)))

        # ResNet
        for i in range(res_blocks):
            block = Block(in_channels=block_channels[-1], out_channels=block_channels[-1], device=device)
            squeeze_and_excitation = SqueezeAndExcitation(channels=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((block, squeeze_and_excitation)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            block = Block(in_channels=block_channels[i+1], out_channels=block_channels[i], device=device)
            squeeze_and_excitation = SqueezeAndExcitation(channels=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((block, squeeze_and_excitation)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=in_channels, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):

        X = []

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)

        x = self.ci(x)
        x = F.relu(x)

        for (block, sae) in self.D_blocks:
            X.append(x)
            x = torch.nn.functional.max_pool2d(x, 2)
            x = block(x)
            x = sae(x)

        for (block, sae) in self.R_blocks:
            x = block(x)
            x = sae(x)

        for i, (block, sae) in enumerate(self.U_blocks):
            x = block(x)
            x = sae(x)

            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            x = x + X[-i-1]

        x = self.cf(x)
        return x
