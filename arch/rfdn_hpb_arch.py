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

def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(
        inChannels, outChannels, kernelSize,
        padding=(kernelSize//2), bias=bias)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True):
        super().__init__()

        self.reduction = defaultConv(
            inChannel, outChannel//2, kernelSize, bias)
        self.expansion = defaultConv(
            outChannel//2, inChannel, kernelSize, bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res

        return x


class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)
        self.conv1 = defaultConv(2*inChannel, 2*outChannel, kernelSize=1)
        self.conv3 = defaultConv(2*inChannel, outChannel, kernelSize=3)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):

        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x

# High Filter Module
class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        
        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size = self.k, stride = self.k),
            nn.Upsample(scale_factor = self.k, mode = 'nearest'),
        )

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)

class HPB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.hfm = HFM()
        self.arfb1 = ARFB(inChannel, outChannel, reScale)
        self.arfb2 = ARFB(inChannel, outChannel, reScale)
        self.arfb3 = ARFB(inChannel, outChannel, reScale)
        self.arfbShare = ARFB(inChannel, outChannel, reScale)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.se = SELayer(inChannel)
        self.conv1 = defaultConv(2*inChannel, outChannel, kernelSize=1 )
    def forward(self,x):
        ori = x
        x = self.arfb1(x)
        x = self.hfm(x)
        x = self.arfb2(x)
        x_share = F.interpolate(x,scale_factor=0.5)
        for _ in range(5):
            x_share = self.arfbShare(x_share)
        x_share = self.upsample(x_share)

        x = torch.cat((x_share,x),1)
        x = self.conv1(x)
        x = self.se(x)
        x = self.arfb3(x)
        x = ori+x
        return x
        

class Config():
    lamRes = torch.nn.Parameter(torch.ones(1))
    lamX = torch.nn.Parameter(torch.ones(1))

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class RFDN_hpb(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(RFDN_hpb, self).__init__()
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

        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        # RFDB_block_f = functools.partial(RFDB, in_channels=num_feat, conv=self.conv, p=p)
        # RFDB_trunk = make_layer(RFDB_block_f, num_block)
        self.B1 = HPB(inChannel=num_feat, outChannel=num_feat, reScale=self.adaptiveWeight)
        self.B2 = HPB(inChannel=num_feat, outChannel=num_feat, reScale=self.adaptiveWeight)
        self.B3 = HPB(inChannel=num_feat, outChannel=num_feat, reScale=self.adaptiveWeight)
        self.B4 = HPB(inChannel=num_feat, outChannel=num_feat, reScale=self.adaptiveWeight)
        # self.B5 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        # self.B6 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
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
        # out_B5 = self.B5(out_B4)
        # out_B6 = self.B6(out_B5)
        # out_B7 = self.B7(out_B6)
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1)
        out_B = self.c1(trunk.permute(0, 2, 3, 1))
        out_B = self.GELU(out_B.permute(0, 3, 1, 2))
        # print(out_B.shape)
        out_lr = self.c2(out_B) + out_fea

        # output = self.c3(out_lr)
        output = self.upsampler(out_lr)

        return output

if __name__ == '__main__':
    upscale = 4
    dec_rate = 0.9
    model = RFDN_hpb(
        num_in_ch=3,
        num_feat=50,
        num_block=4,
        num_out_ch=3,
        upscale=4,
        conv='BSconvU',
        upsampler= 'pixelshuffledirect')
        # p=0.25,
        # dec_rate=0.9)
    print(model)
    x = torch.randn((1, 3, 256, 256))
    x = model(x)
    print(x.shape)