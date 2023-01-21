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



############# Previously used models ##############

class FirstModel(nn.Module):
    """Basic UNet diffusion model with ResNet layers.
    This model ignores the timestep.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, block_channels=[8, 16, 32, 64, 128]):
        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()      # downsampling blocks
        self.R_blocks = nn.ModuleList()      # ResNet blocks
        self.U_blocks = nn.ModuleList()      # upsampling blocks

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=block_channels[0], kernel_size=3, padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

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

        x = self.cf(x)
        return x


############ UNetResNet variations #############

class UNetResNet(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    The timestep encoding is concatenated to the spatial activations after a first convolution.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

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

        x = self.cf(x)
        return x


class UNetResNetSum(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    The timestep encoding is summed to the spatial activations after a first convolution.
    """

    def __init__(self, img_shape, device='cpu', res_blocks=3, block_channels=[8, 16, 32, 64, 128]):
        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=block_channels[0], kernel_size=3, padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

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


class UNetResNetK2(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    Similar to ``UNetResNet``, but the kernel size of the first layer of each down- and upsampling block
    has been reduced to 2x2.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

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

        x = self.cf(x)
        return x


class UNetResNetFC(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    It is ``UNetResNet`` with an additional intermediate convolutional layer before the final one,
    with half the number of channels in its output compared to its input.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3,
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # final convolution layers
        self.cf1 = nn.Conv2d(in_channels=block_channels[0], out_channels=block_channels[0]//2, kernel_size=3, padding=1, device=device)
        self.cf2 = nn.Conv2d(in_channels=block_channels[0]//2, out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

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

        x = self.cf1(x)
        x = self.cf2(x)
        return x


class UNetResNetFixed(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    In constrast to ``UNetResNet``, the downsampling is fixed to maxpooling and the upsampling is done by bilinear interpolation.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                      # downsampling blocks
        self.R_blocks = nn.ModuleList()                      # ResNet blocks
        self.U_blocks = nn.ModuleList()                      # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels,
                          spatial_dimensions=x.shape[-2:], device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x=torch.nn.functional.max_pool2d(x, 2)

            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

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

            x = torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear')

            x = c2(torch.concat([x,X[-i-2]],1))
            x = bn2(x)
            x = F.relu(x)

        x = self.cf(x)
        return x


############# Attention Models #############

from einops import rearrange
"""https://huggingface.co/blog/annotated-diffusion"""


class LinearAttentionBlock(nn.Module):
    """Linear attention module for CNN, as implemented in
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

    This version is more efficient than standard attention, because its time and memory requirements
    scale linearly with the input size rahter than quadratically.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)  # Softmax is applied before the dot product
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # Dot product between keys and values

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out)


class AttentionBlock(nn.Module):
    """Standard attention module for CNN, with an implementation more compliant with the one of the paper
    'Attention is all you need'
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale
        #v = v / (h * w)

        context = torch.einsum('b h d n, b h d m -> b h n m', q, k)  # Dot product between queries and keys

        #context = (context-context.amax(dim=-1)).softmax(dim=-1)  # Softmax is applied after the dot product
        context = context.softmax(dim=-1)

        out = torch.einsum('b h n m, b h d m -> b h n d', context, v)

        out = rearrange(out, 'b h (x y) c -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out)


class UNetResNetLinearAttention(nn.Module):
    """UNet diffusion model with ResNet layers, positional encoding for the timestep and linear attention.
    The downsampling is fixed to maxpooling and the upsampling is done by bilinear interpolation.
    Linear attention is used.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                     # downsampling blocks
        self.R_blocks = nn.ModuleList()                     # ResNet blocks
        self.U_blocks = nn.ModuleList()                     # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            att = LinearAttentionBlock(dim=block_channels[i+1])
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,att)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            att = LinearAttentionBlock(dim=block_channels[i])
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,att)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels, spatial_dimensions=x.shape[-2:], 
                          device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2, att in self.D_blocks:
            x = torch.nn.functional.max_pool2d(x, 2)

            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

            x = att(x)

            X.append(x)

        for c1, bn1, c2, bn2 in self.R_blocks:
            xr = c1(x)
            xr = bn1(xr)
            xr = F.relu(xr)

            xr = c2(xr)
            xr = bn2(xr+x)
            x = F.relu(xr)

        for i, (c1, bn1, c2, bn2, att) in enumerate(self.U_blocks):
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear')

            x = c2(torch.concat([x,X[-i-2]],1))
            x = bn2(x)
            x = F.relu(x)

            x = att(x)

        x = self.cf(x)
        return x


class UNetResNetQuadraticAttention(nn.Module):
    """UNet diffusion model with ResNet layers, positional encoding for the timestep and attention.
    The downsampling is fixed to maxpooling and the upsampling is done by bilinear interpolation.
    Quadratic attention is used.
    """
    def __init__(self, img_shape, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[32, 64, 128, 256, 512]):
        if temp_encoding_initial_channels>block_channels[0]:
            raise ValueError('`temp_encoding_initial_channels` must be smaller than `block_channels[0]s`')

        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()                     # downsampling blocks
        self.R_blocks = nn.ModuleList()                     # ResNet blocks
        self.U_blocks = nn.ModuleList()                     # upsampling blocks
        self.block_channels = block_channels    # for summing with positional encoding of t

        self.temp_encoding_initial_channels = temp_encoding_initial_channels
        self.spatial_encoding_initial_channels = block_channels[0] - temp_encoding_initial_channels

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=self.spatial_encoding_initial_channels, kernel_size=3, 
                            padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            att = AttentionBlock(dim=block_channels[i+1])
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,att)))

        # ResNet
        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[-1], out_channels=block_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            c1 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i], device=device)
            att = AttentionBlock(dim=block_channels[i])
            self.U_blocks.append(nn.ModuleList((c1,bn1,c2,bn2,att)))

        # final convolution layer
        self.cf = nn.Conv2d(in_channels=block_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)

    def forward(self, x, t):
        X = []

        x = self.ci(x)
        x = F.relu(x)

        tp = pos_encoding(t=t, channels=self.temp_encoding_initial_channels, spatial_dimensions=x.shape[-2:], 
                          device=self.device)
        x = torch.cat([x, tp], dim=1)
        X.append(x)

        for c1, bn1, c2, bn2, att in self.D_blocks:
            x = torch.nn.functional.max_pool2d(x, 2)

            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

            x = att(x)

            X.append(x)

        for c1, bn1, c2, bn2 in self.R_blocks:
            xr = c1(x)
            xr = bn1(xr)
            xr = F.relu(xr)

            xr = c2(xr)
            xr = bn2(xr+x)
            x = F.relu(xr)

        for i, (c1, bn1, c2, bn2, att) in enumerate(self.U_blocks):
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear')

            x = c2(torch.concat([x,X[-i-2]],1))
            x = bn2(x)
            x = F.relu(x)

            x = att(x)

        x = self.cf(x)
        return x


############# Squeeze and Excitation #############

class GroupedConvBlock(nn.Module):
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


class SqueezeAndExcitationBlock(nn.Module):
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


class SqueezeAndExcitationModel(nn.Module):
    """Uses X blocks with Gruped Convolution and Squeeze-and-Excitation."""
    def __init__(self, in_channels=3, device='cpu', res_blocks=3, temp_encoding_initial_channels=10,
                 block_channels=[128, 256, 512, 1024, 2048]):
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
            block = GroupedConvBlock(in_channels=block_channels[i], out_channels=block_channels[i+1], groups=16, device=device)
            squeeze_and_excitation = SqueezeAndExcitationBlock(channels=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((block, squeeze_and_excitation)))

        # ResNet
        for i in range(res_blocks):
            block = GroupedConvBlock(in_channels=block_channels[-1], out_channels=block_channels[-1], groups=16, device=device)
            squeeze_and_excitation = SqueezeAndExcitationBlock(channels=block_channels[-1], device=device)
            self.R_blocks.append(nn.ModuleList((block, squeeze_and_excitation)))

        # upsampling
        for i in reversed(range(len(block_channels)-1)):
            block = GroupedConvBlock(in_channels=block_channels[i+1], out_channels=block_channels[i], groups=16, device=device)
            squeeze_and_excitation = SqueezeAndExcitationBlock(channels=block_channels[i], device=device)
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
