import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encodings import pos_encoding


class UNetBottleneckResidual(nn.Module):
    """UNet diffusion model with residual bottleneck blocks.

    The model takes as input the noisy image 'x' and the timesetp 't' and returns the noise at that timestep.

    The general structure is the same of a UNet: downsampling, intermediate and upsampling parts.
    Each part is a stack of residual bottleneck blocks.

    The timestep is processed using positional encoding into `timestep_encoding_initial_channels` channels.
    These channels are concatenated with the input image and passed through an initial convolutional layer.
    The resulting activations are given to the UNet.

    `block_channels` specifies the number of channels of each downsampling and upsampling block.
    Each downnsampling block halves the spatial dimensions, through MaxPool.
    Each upsampling block doubles the spatial dimensions through bilinear interpolation.
    The intermediate part is a classig ResNet.

    In the downsampling and upsampling parts, the blocks are not exactly bottleneck residual.
    For the downsampling:
     - the skip connections are not present (it is not a residual layer, but only a bottleneck layer)
    For the upsampling:
     - the skip connections come only from the corresponding downsampling activations, and not from the input of that block (it is not a proper residual layer)
     - the third convolution in the block uses a 3x3 kernel instead of a 1x1.
    We found this configuration to be more effective in practice than the non bottleneck configuration.

    Parameters
    ----------
    img_shape : 3-tuple of int
        `(num_channels, height, width)`
    device : torch.device, optional
        by default 'cpu'
    res_blocks : int, optional
        number of intermediate ResNet blocks, by default 3
    timestep_encoding_initial_channels : int, optional
        number of channels for the positional encoding of the timestep in the input layer, by default 10
    block_channels : list, optional
        number of channels in each down- and upsampling block, by default [32, 64, 128, 256, 512]
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, timestep_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if timestep_encoding_initial_channels>block_channels[0]:
            raise ValueError('`timestep_encoding_initial_channels` must be smaller than `block_channels[0]`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.timestep_encoding_initial_channels = timestep_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth+timestep_encoding_initial_channels, out_channels=block_channels[0]*4, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            bn1 = nn.BatchNorm2d(num_features=block_channels[i]*4, device=device)
            c1 = nn.Conv2d(in_channels=block_channels[i]*4, out_channels=block_channels[i+1], kernel_size=1, padding=0, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn3 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c3 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1]*4, kernel_size=1, padding=0, device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,c3,bn3)))

        # Intermediate ResNet
        for i in range(res_blocks):
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1]*4, device=device)
            c1 = nn.Conv2d(in_channels=block_channels[-1]*4, out_channels=block_channels[-1], kernel_size=1, padding=0, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn3 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c3 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1]*4, kernel_size=1, padding=0, device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,c3,bn3)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1]*4, device=device)
            c1 = nn.Conv2d(in_channels=block_channels[i+1]*4, out_channels=block_channels[i+1], kernel_size=1, padding=0, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn3 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c3 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i]*4, kernel_size=3, padding=1, device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,c3,bn3)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0]*4, out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        tp = pos_encoding(t=t, channels=self.timestep_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)
        x = self.ci(x)
        x = F.relu(x)

        X.append(x)

        for (c1,bn1,c2,bn2,c3,bn3) in self.D_blocks:
            x=torch.nn.functional.max_pool2d(x, 2)

            x = bn1(x)
            x = F.relu(x)
            x = c1(x)

            x = bn2(x)
            x = F.relu(x)
            x = c2(x)

            x = bn3(x)
            x = F.relu(x)
            x = c3(x)

            X.append(x)

        for (c1,bn1,c2,bn2,c3,bn3) in self.R_blocks:
            xr = bn1(x)
            xr = F.relu(xr)
            xr = c1(xr)

            xr = bn2(xr)
            xr = F.relu(xr)
            xr = c2(xr)

            xr = bn3(xr)
            xr = F.relu(xr)
            x = c3(xr)+x

        for i, (c1,bn1,c2,bn2,c3,bn3) in enumerate(self.U_blocks):
            x = bn1(x)
            x = F.relu(x)
            x = c1(x)

            x = bn2(x)
            x = F.relu(x)
            x = c2(x)

            x = bn3(x)
            x = F.relu(x)
            x = c3(x)

            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')

            x = x + X[-i-2]

        x = self.cf(x)
        return x
