import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Basic model for predicting the timestep t of a noisy image.
    This model simply consists in a sequence of downsampling convolutions and then a final linear layer.
    """
    def __init__(self, img_shape,  device='cpu', block_channels=[8, 16, 32, 64, 128, 256]):
        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        self.D_blocks = nn.ModuleList()      # downsampling blocks

        # initial convolution layer
        self.ci = nn.Conv2d(in_channels=img_depth, out_channels=block_channels[0], kernel_size=3, padding=1, device=device)

        # downsampling
        for i in range(len(block_channels)-1):
            c1 = nn.Conv2d(in_channels=block_channels[i], out_channels=block_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_channels[i+1], out_channels=block_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_channels[i+1], device=device)
            self.D_blocks.append(nn.ModuleList((c1,bn1,c2,bn2)))

        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # final linear layers
        self.l1 = nn.Linear(in_features=block_channels[-1], out_features=block_channels[-2], device=device)
        self.l2 = nn.Linear(in_features=block_channels[-2], out_features=1, device=device)
        

    def forward(self, x):
        x = self.ci(x)
        x = F.relu(x)

        for c1, bn1, c2, bn2 in self.D_blocks:
            x = c1(x)
            x = bn1(x)
            x = F.relu(x)

            x = c2(x)
            x = bn2(x)
            x = F.relu(x)

        x = self.gap(x)
        x = x.reshape(x.shape[0:2])
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)

        return x
