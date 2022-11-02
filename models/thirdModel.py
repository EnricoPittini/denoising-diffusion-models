import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.encodings import pos_encoding
from einops import rearrange

"""https://huggingface.co/blog/annotated-diffusion"""


class LinearAttention(nn.Module):
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


class Attention(nn.Module):
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


class ThirdModelV1(nn.Module):
    """UNet diffusion model with ResNet layers, positional encoding for the timestep and linear attention.
    The downsampling is fixed to maxpooling and the upsampling is done by bilinear interpolation.
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
            att = LinearAttention(dim=block_channels[i+1])
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
            att = LinearAttention(dim=block_channels[i])
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


class ThirdModelV2(nn.Module):
    """UNet diffusion model with ResNet layers, positional encoding for the timestep and attention.
    The downsampling is fixed to maxpooling and the upsampling is done by bilinear interpolation.
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
            att = Attention(dim=block_channels[i+1])
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
            att = Attention(dim=block_channels[i])
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
