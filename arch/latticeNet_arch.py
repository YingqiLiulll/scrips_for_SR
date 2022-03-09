from basicsr.utils.registry import ARCH_REGISTRY
from torch.autograd import Variable
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import basicsr.archs.Blocks as Blocks
import basicsr.archs.Upsamplers as Upsamplers
import math





def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## add SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## add SEResBlock
class SEResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x

        return res


### Lattice网络
def make_model(args, parent=False):
    return LatticeNet_split(args)

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
    def __init__(self, in_ch):
        super(YQBlock, self).__init__()
        if in_ch % 2 != 0:
            assert ValueError("odd in_ch!")
        conv = Blocks.BSConvU
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': 0.25}
        self.in_ch = in_ch
        rc = in_ch // 2
        self.head_conv = conv(in_ch, in_ch, kernel_size=3, with_ln=False, **BSConvS_kwargs)
        self.act = nn.LeakyReLU(0.05)
        self.sigmoid = nn.Sigmoid()

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
        self.lower_linear = nn.Linear(rc, in_ch)
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

        # lower of lower
        self.lower_of_lower_linear = nn.Linear(rc, in_ch)
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
        x = self.head_conv(input)
        head_out = self.act(x+input)

        # Channel Split
        upper, lower = torch.split(head_out, (self.in_ch//2, self.in_ch//2),dim=1)

        # upper
        upper = self.upper_linear(upper.permute(0, 2, 3, 1))
        upper = self.upper_dw(upper.permute(0, 3, 1, 2))
        upper_attn = self.sigmoid(upper)

        # lower
        lower_0 = self.lower_linear(lower.permute(0, 2, 3, 1))
        lower_0 = self.lower_dw(lower_0.permute(0, 3, 1, 2))

        lower_1 = self.lower_of_lower_linear(lower.permute(0, 2, 3, 1))
        lower_1 = self.lower_of_lower_dw(lower_1.permute(0, 3, 1, 2))

        lower_out = lower_1 + lower_0
        out = torch.mul(lower_out, upper_attn)
        return out


class LatticeBlock(nn.Module):
    def __init__(self, nFeat):
        super(LatticeBlock, self).__init__()

        self.D3 = nFeat
        # self.d = nDiff
        # self.s = nFeat_slice

        # block_0 = []
        # block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        # block_0.append(nn.LeakyReLU(0.05))
        # block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        # block_0.append(nn.LeakyReLU(0.05))
        # block_0.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        # block_0.append(nn.LeakyReLU(0.05))
        # self.conv_block0 = nn.Sequential(*block_0)
        self.conv_block0 = YQBlock(nFeat)

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        # block_1 = []
        # block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        # block_1.append(nn.LeakyReLU(0.05))
        # block_1.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        # block_1.append(nn.LeakyReLU(0.05))
        # block_1.append(nn.Conv2d(nFeat-nDiff, nFeat, kernel_size=3, padding=1, bias=True))
        # block_1.append(nn.LeakyReLU(0.05))
        # self.conv_block1 = nn.Sequential(*block_1)
        self.conv_block1 = YQBlock(nFeat)

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


@ARCH_REGISTRY.register()
class LatticeNet_split(nn.Module):
    def __init__(self, n_feats=50, scale=4, conv='DepthWiseConv',p=0.25, upsampler='pixelshuffledirect',num_out_ch=3):
        super(LatticeNet_split, self).__init__()
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

        rgb_range = 1.
        # n_feats = args.n_feats
        # scale = args.scale[0]

        nFeat = n_feats
        # nDiff = 16
        # nFeat_slice = 4
        nChannel = 3

        # RGB mean for DF2K
        rgb_mean = (0.4029, 0.4484, 0.4680)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        # self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        self.conv1 = self.conv(nChannel, nFeat, kernel_size=3, **kwargs)
        self.conv2 = self.conv(nFeat, nFeat, kernel_size=3, **kwargs)

        # define body module
        self.body_unit1 = LatticeBlock(nFeat)
        self.body_unit2 = LatticeBlock(nFeat)
        self.body_unit3 = LatticeBlock(nFeat)
        self.body_unit4 = LatticeBlock(nFeat)

        self.T_tdm1 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())
        self.L_tdm1 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())

        self.T_tdm2 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())
        self.L_tdm2 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())

        self.T_tdm3 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())
        self.L_tdm3 = nn.Sequential(
            # nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU())

        # define tail module
        modules_tail = [self.conv(n_feats, n_feats, kernel_size=3, **kwargs),
                        self.conv(n_feats, 3 * (scale ** 2), kernel_size=3, **kwargs),
                        nn.PixelShuffle(scale)]
        self.tail = nn.Sequential(*modules_tail)

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.conv1(x)
        x = self.conv2(x)

        res1 = self.body_unit1(x)
        res2 = self.body_unit2(res1)
        res3 = self.body_unit3(res2)
        res4 = self.body_unit4(res3)

        T_tdm1 = self.T_tdm1(res4.permute(0, 2, 3, 1))
        L_tdm1 = self.L_tdm1(res3.permute(0, 2, 3, 1))
        out_TDM1 = torch.cat((T_tdm1, L_tdm1), 3)

        T_tdm2 = self.T_tdm2(out_TDM1)
        L_tdm2 = self.L_tdm2(res2.permute(0, 2, 3, 1))
        out_TDM2 = torch.cat((T_tdm2, L_tdm2), 3)

        T_tdm3 = self.T_tdm3(out_TDM2)
        L_tdm3 = self.L_tdm3(res1.permute(0, 2, 3, 1))
        out_TDM3 = torch.cat((T_tdm3, L_tdm3), 3)
        out_TDM3 = out_TDM3.permute(0, 3, 1, 2)

        res = out_TDM3 + x
        out = self.tail(res)

        x = self.add_mean(out)

        return x


    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') >= 0:
    #                     print('Replace pre-trained upsampler to new one...')
    #                 else:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))

    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    upscale = 4
    # dec_rate = 0.9
    model = LatticeNet_split(
        n_feats=50,
        scale=4,
        )
        # conv='BSconvU',
        # upsampler= 'pixelshuffledirect',
        # p=0.25,
        # dec_rate=0.9)
    print(model)
    x = torch.randn((1, 3, 256, 256))
    x = model(x)
    print(x.shape)