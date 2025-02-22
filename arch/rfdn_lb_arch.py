'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Blocks as Blocks
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY


def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

class CoffConv(nn.Module):
    def __init__(self, num_fea):
        super(CoffConv, self).__init__()
        self.AdaptiveAvgPool2d =  nn.AdaptiveAvgPool2d(1)
        self.upper_branch = nn.Sequential(
            nn.Linear(num_fea, num_fea // 16),
            # nn.Conv2d(num_fea, num_fea // 16, 1, 1, 0),
            nn.GELU(),
            nn.Linear(num_fea // 16, num_fea),
            # nn.Conv2d(num_fea // 16, num_fea, 1, 1, 0),
            nn.GELU(),
            nn.Sigmoid()
        )
        
        self.std = std
        self.lower_branch = nn.Sequential(
            nn.Linear(num_fea, num_fea // 16),
            # nn.Conv2d(num_fea, num_fea // 16, 1, 1, 0),
            nn.GELU(),
            nn.Linear(num_fea // 16, num_fea),
            # nn.Conv2d(num_fea // 16, num_fea, 1, 1, 0),
            nn.GELU(),
            nn.Sigmoid()
        )

    def forward(self, fea):
        fea_1 = self.AdaptiveAvgPool2d(fea).permute(0, 2, 3, 1)
        upper = self.upper_branch(fea_1)
        upper = upper.permute(0, 3, 1, 2)
        lower = self.std(fea)
        lower = self.lower_branch(lower.permute(0, 2, 3, 1))
        lower = lower.permute(0, 3, 1, 2)

        out = torch.add(upper, lower) / 2
        
        return out


class LBlock(nn.Module):
    def __init__(self, num_fea):
        super(LBlock, self).__init__()
        self.H_conv = nn.Sequential(
            nn.Conv2d(num_fea, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, num_fea, 3, 1, 1),
            nn.LeakyReLU(0.05),
        )

        self.A1_coff_conv = CoffConv(num_fea)
        self.B1_coff_conv = CoffConv(num_fea)

        self.G_conv = nn.Sequential(
            nn.Conv2d(num_fea, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, num_fea, 3, 1, 1),
            nn.LeakyReLU(0.05),
        )

        self.A2_coff_conv = CoffConv(num_fea)
        self.B2_coff_conv = CoffConv(num_fea)

        self.fuse = nn.Linear(num_fea*2, num_fea)
        # self.fuse = nn.Conv2d(num_fea*2, num_fea, 1, 1, 0)


    def forward(self, x):
        H = self.H_conv(x)
        A1 = self.A1_coff_conv(H)
        P1 = x + A1*H
        B1 = self.B1_coff_conv(x)
        Q1 = H + B1*x

        G = self.G_conv(P1)
        B2 = self.B2_coff_conv(G)
        Q2 = Q1 + B2*G
        A2 = self.A2_coff_conv(Q1)
        P2 = G + Q1*A2

        out = torch.cat([P2, Q2], dim=1).permute(0, 2, 3, 1)
        out = self.fuse(out).permute(0, 3, 1, 2)

        return out


class BFModule(nn.Module):
    def __init__(self, num_fea):
        super(BFModule, self).__init__()
        # self.conv4 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        # self.conv3 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        # self.fuse43 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(num_fea, num_fea//2, 1, 1,0)        
        # self.fuse32 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        # self.conv1 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        self.conv4 = nn.Linear(num_fea, num_fea//2)
        self.conv3 = nn.Linear(num_fea, num_fea//2)
        self.fuse43 = nn.Linear(num_fea, num_fea//2)
        self.conv2 = nn.Linear(num_fea, num_fea//2)
        self.fuse32 = nn.Linear(num_fea, num_fea//2)
        self.conv1 = nn.Linear(num_fea, num_fea//2)

        self.act = nn.GELU(inplace=True)

    # def forward(self, x_list):
    #     dr_1 = self.conv4(x_list[3].permute(0, 2, 3, 1))
    #     H4 = self.act(dr_1.permute(0, 3, 1, 2))
    #     dr_2 = self.conv3(x_list[2].permute(0, 2, 3, 1))
    #     H3_half = self.act(dr_2.permute(0, 3, 1, 2))
    #     H3 = self.fuse43(torch.cat([H4, H3_half], dim=1).permute(0, 2, 3, 1))
    #     H3 = H3.permute(0, 3, 1, 2)
    #     dr_3 = self.conv2(x_list[2].permute(0, 2, 3, 1))      
    #     H2_half = self.act(dr_3.permute(0, 3, 1, 2))
    #     H2 = self.fuse32(torch.cat([H3, H2_half], dim=1).permute(0, 2, 3, 1))
    #     H2 = H2.permute(0, 3, 1, 2)
    #     H1_half = self.act(self.conv1(x_list[0]).permute(0, 2, 3, 1))
    #     H1 = torch.cat([H2, H1_half], dim=1)

    def forward(self, x_list):
        H4 = self.act(self.conv4(x_list[3].permute(0, 2, 3, 1)))
        H3_half = self.act(self.conv3(x_list[2].permute(0, 2, 3, 1)))
        H3 = self.fuse43(torch.cat([H4, H3_half], dim=3))      
        H2_half = self.act(self.conv2(x_list[1].permute(0, 2, 3, 1)))
        H2 = self.fuse32(torch.cat([H3, H2_half], dim=3))
        H1_half = self.act(self.conv1(x_list[0].permute(0, 2, 3, 1)))
        H1 = torch.cat([H2, H1_half], dim=3)
        H1 = H1.permute(0, 3, 1, 2)

        return H1

class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Linear(num_feat, f)
        self.conv_f = nn.Linear(f, f)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Linear(f, num_feat)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        x = input.permute(0, 2, 3, 1)
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_.permute(0, 3, 1, 2))
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3.permute(0, 2, 3, 1) + cf))
        m = self.sigmoid(c4.permute(0, 3, 1, 2))

        return input * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, conv=nn.Conv2d, p=0.25):
        super(RFDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Linear(in_channels, self.dc)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Linear(self.remaining_channels, self.dc)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Linear(self.remaining_channels, self.dc)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Linear(self.dc * 4, in_channels)
        self.esa = ESA(in_channels, conv)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input.permute(0, 2, 3, 1)))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1.permute(0, 2, 3, 1)))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2.permute(0, 2, 3, 1)))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4.permute(0, 2, 3, 1)], dim=3)
        out = self.c5(out).permute(0, 3, 1, 2)
        out_fused = self.esa(out)

        return out_fused


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class RFDN_LB(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(RFDN_LB, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        print(conv)
        if conv == 'DepthWiseConv':
            self.conv = Blocks.DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = Blocks.BSConvU
        elif conv == 'BSConvS':
            self.conv = Blocks.BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch, num_feat, kernel_size=3, **kwargs)

        # RFDB_block_f = functools.partial(RFDB, in_channels=num_feat, conv=self.conv, p=p)
        # RFDB_trunk = make_layer(RFDB_block_f, num_block)
        self.B1 = LBlock(num_feat)
        self.B2 = LBlock(num_feat)
        self.B3 = LBlock(num_feat)
        self.B4 = LBlock(num_feat)
        self.B5 = LBlock(num_feat)
        self.B6 = LBlock(num_feat)
        # self.B7 = RFDB(in_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Linear(num_feat * num_block, num_feat)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        # out_B7 = self.B7(out_B6)
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        out_B = self.c1(trunk.permute(0, 2, 3, 1))
        out_B = self.GELU(out_B.permute(0, 3, 1, 2))
        # print(out_B.shape)
        out_lr = self.c2(out_B) + out_fea

        # output = self.c3(out_lr)
        output = self.upsampler(out_lr)

        return output

# if __name__ == '__main__':
#     upscale = 4
#     dec_rate = 0.9
#     model = RFDN_LB(
#         num_in_ch=3,
#         num_feat=50,
#         num_block=6,
#         num_out_ch=3,
#         upscale=4)
#         # conv='BSconvU',
#         # upsampler= 'pixelshuffledirect',
#         # p=0.25,
#         # dec_rate=0.9)
#     print(model)
#     x = torch.randn((1, 3, 256, 256))
#     x = model(x)
#     print(x.shape)