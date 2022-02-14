import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ESA(nn.Module):
    def __init__(self, num_feat=50):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1, 1, 0)
        self.conv_f = nn.Conv2d(f, f, 1, 1, 0)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = nn.Conv2d(f, f, 3, 1, 1)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = nn.Conv2d(f, f, 3, 1, 1)
        self.conv3_ = nn.Conv2d(f, f, 3, 1, 1)
        self.conv4 = nn.Conv2d(f, num_feat, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d =nn.Conv2d(in_channels, self.dc, 1, 1, 0)
        self.c1_r =nn.Conv2d(in_channels, self.rc, 3, 1, 1)
        self.c2_d =nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0)
        self.c2_r =nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1)
        self.c3_d =nn.Conv2d(self.remaining_channels, self.dc, 1, 1, 0)
        self.c3_r =nn.Conv2d(self.remaining_channels, self.rc, 3, 1, 1)

        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1, 1, 0)
        self.pa = PA(in_channels)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.pa(self.c5(out))

        return out_fused


@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4):
        super(RFDN, self).__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.B1 = RFDB(in_channels = num_feat)
        self.B2 = RFDB(in_channels = num_feat)
        self.B3 = RFDB(in_channels = num_feat)
        self.B4 = RFDB(in_channels = num_feat)
        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.05, inplace=True)

        self.c2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.c3 = nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1)
        self.pixel = nn.PixelShuffle(upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c1(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_B = self.lrelu(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.c3(out_lr)
        output = self.pixel(output)

        return output

