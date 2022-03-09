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

class LPA(nn.Module):
    def __init__(self, n_feats,conv='BSConvU'):
        super(LPA, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvU':
            self.conv = Blocks.BSConvU
        f = n_feats // 4
        # self.collapse = conv(n_feats, f, 1)
        self.collapse = nn.Linear(n_feats,f)
        self.squeeze = nn.MaxPool2d(4, 4)
        self.redistribution = self.conv(f, f, kernel_size=3, **kwargs)
        # self.restore = conv(f, n_feats, 1)
        self.restore = nn.Linear(f, n_feats)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        y = self.collapse(x.permute(0, 2, 3, 1))
        y = y.permute(0, 3, 1, 2)
        res = y
        y = self.squeeze(y)
        y = self.redistribution(y)
        y = F.interpolate(y, x.size()[2:], mode='bilinear', align_corners=False)
        y = self.restore((y + res).permute(0, 2, 3, 1))
        y = self.sigmoid(y.permute(0, 3, 1, 2))
        return x * y

class CFB(nn.Module):
    def __init__(self, n_feats,conv='BSConvU'):
        super(CFB, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvU':
            self.conv = Blocks.BSConvU
        self.conv1 = self.conv(n_feats, n_feats, kernel_size=3, **kwargs)
        self.conv2 = self.conv(n_feats, n_feats, kernel_size=3, **kwargs)
        self.conv3 = self.conv(n_feats, n_feats, kernel_size=3, **kwargs)
        # self.fusion = conv(n_feats * 3, n_feats, 1)
        self.fusion = nn.Linear(n_feats * 3, n_feats)
        self.act = nn.LeakyReLU(0.2, True)
        self.att = LPA(n_feats)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1 + x))
        x3 = self.act(self.conv3(x2 + x))
        trunk = torch.cat([x1, x2, x3], 1).permute(0, 2, 3, 1)
        x = self.att(self.fusion(trunk).permute(0, 3, 1, 2) + x)
        return x


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
class RFDN_cascade(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(RFDN_cascade, self).__init__()
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
        self.B1 = CFB(n_feats=num_feat, conv='BSConvU')
        self.B2 = CFB(n_feats=num_feat, conv='BSConvU')
        self.B3 = CFB(n_feats=num_feat, conv='BSConvU')
        self.B4 = CFB(n_feats=num_feat, conv='BSConvU')
        self.B5 = CFB(n_feats=num_feat, conv='BSConvU')
        self.B6 = CFB(n_feats=num_feat, conv='BSConvU')
        # self.B7 = RFDB(in_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Linear(num_feat * num_block, num_feat)
        self.GELU = nn.GELU()

        # self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        self.c2 = LPA(n_feats=num_feat, conv='BSConvU')


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

if __name__ == '__main__':
    upscale = 4
    # dec_rate = 0.9
    model = RFDN_cascade(
        num_in_ch=3,
        num_feat=50,
        num_block=6,
        num_out_ch=3,
        upscale=4,
        conv='BSconvU',
        upsampler= 'pixelshuffledirect')
    print(model)
    x = torch.randn((1, 3, 256, 256))
    x = model(x)
    print(x.shape)