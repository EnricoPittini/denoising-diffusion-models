import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encodings import pos_encoding

class SecondModelSum(nn.Module):
    """Basic UNet diffusion model with ResNet layers
    and positional encoding for the timestep.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, block_channels=[8, 16, 32, 64, 128]):
        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = []                      # downsampling blocks
        self.R_blocks = []                      # ResNet blocks
        self.U_blocks = []                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=block_channels[0], kernel_size=3, padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append((c1,bn1,c2,bn2))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append((c1,bn1,c2,bn2))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append((c1,bn1,c2,bn2))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)


    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=x.shape[1], spatial_dimensions=x.shape[-2:], device=self.device)
        x = x + tp
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

            x = x + pos_encoding(t=t, channels=x.shape[1], spatial_dimensions=x.shape[-2:], device=self.device)

            X.append(x)

        for c1, bn1, c2, bn2 in self.R_blocks:
            xr = c1(x)
            xr = bn1(xr)
            xr = F.relu(xr)

            xr = c2(xr)
            xr = bn2(xr+x)
            x = F.relu(xr)

        for i, (c1, bn1, c2, bn2) in enumerate(self.U_blocks):
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(torch.concat([x,X[-i-2]],1))
            x = bn2(x)
            x = F.relu(x)

            x = x + pos_encoding(t=t, channels=x.shape[1], spatial_dimensions=x.shape[-2:], device=self.device)

        x = self.cf(x)
        return x
