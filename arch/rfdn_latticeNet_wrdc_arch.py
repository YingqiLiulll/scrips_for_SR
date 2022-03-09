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
from torch.nn.parameter import Parameter

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=50):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


## Combination Coefficient
class CC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Linear(channel // reduction, channel),
                # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True
                nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                # nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Linear(channel // reduction, channel),
                # nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):

        # mean
        ca_mean = self.avg_pool(x).permute(0, 2, 3, 1)
        ca_mean = self.conv_mean(ca_mean)
        ca_mean = ca_mean.permute(0, 3, 1, 2)
        # std
        m_batchsize, C, height, width = x.size()
        x_dense = x.view(m_batchsize, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True)
        ca_std = ca_std.view(m_batchsize, C, 1, 1)
        ca_var = self.conv_std(ca_std.permute(0, 2, 3, 1))
        ca_var = ca_var.permute(0, 3, 1, 2)
        # Coefficient of Variation
        # # cv1 = ca_std / ca_mean
        # cv = torch.div(ca_std, ca_mean)
        # ram = self.sigmoid(ca_mean + ca_var)

        cc = (ca_mean + ca_var)/2.0
        return cc


class YQBlock(nn.Module):
    def __init__(self, in_ch, reScale):
        super(YQBlock, self).__init__()
        if in_ch % 2 != 0:
            assert ValueError("odd in_ch!")
        conv = Blocks.BSConvU
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': 0.25}
        self.in_ch = in_ch
        rc = in_ch // 2
        self.head_conv = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch))
        self.mapping = conv(in_ch, in_ch, kernel_size=3, with_ln=False, **BSConvS_kwargs)
        
        self.act = nn.LeakyReLU(0.05)
        self.sigmoid = nn.Sigmoid()
        self.sa = sa_layer(in_ch,groups=5)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

        # upper
        self.upper_linear = nn.Linear(rc, in_ch)
        self.upper_dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=in_ch,
            bias=True,
            padding_mode="zeros",
        )

        # lower
        self.lower_linear = nn.Sequential(
            nn.Linear(rc, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch))
        self.lower_dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=in_ch,
            bias=True,
            padding_mode="zeros",
        )
        self.sa = sa_layer(in_ch, groups=5)

        # lower of lower
        self.lower_of_lower_linear = nn.Sequential(
            nn.Linear(rc, in_ch),
            nn.GELU(),
            nn.Linear(in_ch, in_ch))
        self.lower_of_lower_dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=in_ch,
            bias=True,
            padding_mode="zeros",
        )



    def forward(self, input):
        # SRB
        # print(input.shape)
        x = self.head_conv(input.permute(0, 2, 3, 1))
        x = self.mapping(x.permute(0, 3, 1, 2))
        x_act = self.act(x)
        head_out = self.lamX * self.sa(x_act)
        head_out = self.lamRes * input + head_out


        # Channel Split
        upper, lower = torch.split(head_out, (self.in_ch//2, self.in_ch//2),dim=1)

        # upper
        upper = self.upper_linear(upper.permute(0, 2, 3, 1))
        upper = self.upper_dw(upper.permute(0, 3, 1, 2))
        upper_attn = self.sigmoid(upper)

        # lower
        lower_0 = self.lower_linear(lower.permute(0, 2, 3, 1))
        lower_0 = self.lower_dw(lower_0.permute(0, 3, 1, 2))
        lower_0 = self.sa(lower_0)

        lower_1 = self.lower_of_lower_linear(lower.permute(0, 2, 3, 1))
        lower_1 = self.lower_of_lower_dw(lower_1.permute(0, 3, 1, 2))

        lower_out = self.lamRes * lower_1 + self.lamX * lower_0
        out = torch.mul(lower_out, upper_attn)
        return out


class LatticeBlock(nn.Module):
    def __init__(self, nFeat, reScale):
        super(LatticeBlock, self).__init__()

        self.D3 = nFeat
        self.conv_block0 = YQBlock(nFeat, reScale)

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)
        
        self.conv_block1 = YQBlock(nFeat, reScale)
        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)
        self.compress = nn.Linear(2 * nFeat, nFeat)
        # self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # analyse unit
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        # synthes_unit
        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        # out = torch.cat((p3z, q3z), 1)
        # out = self.compress(out)
        out = torch.cat((p3z, q3z), 1).permute(0, 2, 3, 1)
        out = self.compress(out)
        out = out.permute(0, 3, 1, 2)

        return out

class Config():
    lamRes = torch.nn.Parameter(torch.ones(1))
    lamX = torch.nn.Parameter(torch.ones(1))

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
class RFDN_latticeNet(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(RFDN_latticeNet, self).__init__()
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
        self.B1 = LatticeBlock(num_feat, reScale=self.adaptiveWeight)
        self.B2 = LatticeBlock(num_feat, reScale=self.adaptiveWeight)
        self.B3 = LatticeBlock(num_feat, reScale=self.adaptiveWeight)
        self.B4 = LatticeBlock(num_feat, reScale=self.adaptiveWeight)
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

# if __name__ == '__main__':
#     upscale = 4
#     dec_rate = 0.9
#     model = RFDN_latticeNet(
#         num_in_ch=3,
#         num_feat=50,
#         num_block=4,
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