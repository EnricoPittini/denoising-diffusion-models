import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstModel(nn.Module):

    def __init__(self, img_shape, device='cpu'):
        super().__init__()
        img_depth = img_shape[-3]
        self.device = device

        block_out_channels=[8,16,32,64,128]
        res_blocks=3

        self.D_blocks=[]
        self.R_blocks=[]
        self.U_blocks=[]

        self.ci=nn.Conv2d(in_channels=img_depth, out_channels=block_out_channels[0], kernel_size=3, padding=1, device=device)

        for i in range(len(block_out_channels)-1):
            c1 = nn.Conv2d(in_channels=block_out_channels[i], out_channels=block_out_channels[i+1], kernel_size=3, padding=1, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_out_channels[i+1], device=device)
            c2 = nn.Conv2d(in_channels=block_out_channels[i+1], out_channels=block_out_channels[i+1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_out_channels[i+1], device=device)
            self.D_blocks.append((c1,bn1,c2,bn2))

        for i in range(res_blocks):
            c1 = nn.Conv2d(in_channels=block_out_channels[-1], out_channels=block_out_channels[-1], kernel_size=3, padding=1, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_out_channels[-1], device=device)
            c2 = nn.Conv2d(in_channels=block_out_channels[-1], out_channels=block_out_channels[-1], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_out_channels[-1], device=device)
            self.R_blocks.append((c1,bn1,c2,bn2))


        for i in reversed(range(len(block_out_channels)-1)):
            c1 = nn.ConvTranspose2d(in_channels=block_out_channels[i+1], out_channels=block_out_channels[i], kernel_size=2, padding=0, stride=2, device=device)
            bn1 = nn.BatchNorm2d(num_features=block_out_channels[i], device=device)
            c2 = nn.Conv2d(in_channels=block_out_channels[i+1], out_channels=block_out_channels[i], kernel_size=3, padding=1, device=device)
            bn2 = nn.BatchNorm2d(num_features=block_out_channels[i], device=device)
            self.U_blocks.append((c1,bn1,c2,bn2))

        self.cf= nn.Conv2d(in_channels=block_out_channels[0], out_channels=img_depth, kernel_size=3, padding=1, device=device)



    def forward(self, x, t):

        X=[]

        x=self.ci(x)
        x=F.relu(x)
        X.append(x)

        #print(x.shape)

        for (c1,bn1,c2,bn2) in self.D_blocks:
            x=c1(x)
            x=bn1(x)
            x=F.relu(x)
            
            #print(x.shape,'D1')

            x=c2(x)
            x=bn2(x)
            x=F.relu(x)

            #print(x.shape,'D2')

            X.append(x)


        for (c1,bn1,c2,bn2) in self.R_blocks:
            xr=c1(x)
            xr=bn1(xr)
            xr=F.relu(xr)
            
            #print(xr.shape,'R1')

            xr=c2(xr)
            xr=bn2(xr+x)
            x=F.relu(xr)

            #print(x.shape,'R2')

        for i,(c1,bn1,c2,bn2) in enumerate(self.U_blocks):
            x=c1(x)
            x=bn1(x)
            x=F.relu(x)
            #print(x.shape,'U1')

            x=c2(torch.concat([x,X[-i-2]],1))
            x=bn2(x)
            x=F.relu(x)

            #print(x.shape,'U2')

        x=self.cf(x)
        #print(x.shape)

        return x
