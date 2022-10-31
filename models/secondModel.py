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


class SecondModelConcat(nn.Module):
    """Basic UNet diffusion model with ResNet layers
    and positional encoding for the timestep.
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


class SecondModelConcatK2(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    The kernel size of the first downsampling layer has been reduced to 2x2 compared to
    ``SecondModelConcat``.
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


class SecondModelConcatFC(nn.Module):
    """Basic UNet diffusion model with ResNet layers and positional encoding for the timestep.
    A final additional convolutional layer is added to ``SecondModelConcat``.
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