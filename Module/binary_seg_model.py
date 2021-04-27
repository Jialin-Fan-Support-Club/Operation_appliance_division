# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size, s=1):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=s)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SE_Module(nn.Module):
    def __init__(self, channel,ratio = 2):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
        self.num=nn.Parameter(torch.ones(1, requires_grad=True))
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)*self.num+x


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()

        self.in_ch = in_ch
        self.mid_mid = out_ch // 7
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])

        self.conv3x3_3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        # self.conv1x1_2 = Conv_block(self.mid_mid, self.mid_mid, [1, 1])
        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.rel = nn.ReLU(inplace=True)
        if self.in_ch > self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.se=SE_Module(out_ch)
    def forward(self, x):
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_1(x1)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5 = self.conv3x3_2_1(x5 + x4)
        x6 = self.conv3x3_3_1(x5 + x6)
        #print(torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1).shape)
        xxx = self.conv1x1_1(torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1))
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x6_2 = xxx[:, self.mid_mid * 6:self.mid_mid * 7, ...]
        x1 = self.conv1x3_2(x1_2)
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5 = self.conv3x3_2_2(x4 + x5_2)
        x6 = self.conv3x3_3_2(x5 + x6_2)
        xx = torch.cat((x0, x1, x2, x3, x4, x5, x6), dim=1)
        xx = self.conv1x1_2(xx)
        xx=self.se(xx)
        if self.in_ch > self.out_ch:
            x = self.short_connect(x)
        return self.rel(xx + x + xxx)


class Conv_down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_down_2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        if self.in_ch == 1:
            self.first = nn.Sequential(
                Conv_block(self.in_ch, self.out_ch, [3, 3]),
                Double_conv(self.out_ch, self.out_ch),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)
        else:
            self.conv = Double_conv(self.in_ch, self.out_ch)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)

    def forward(self, x):
        if self.in_ch == 1:
            x = self.first(x)
            pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
        else:
            x = self.conv(x)
            if self.flage == True:
                pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
            else:
                pool_x = None
        return pool_x, x


class Conv_down2(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch, flage):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down2, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        self.conv = Conv_block(in_ch, self.out_ch, [3, 3], s=2)

    def forward(self, x):
        x = self.conv(x)

        return x


class Conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1

from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)

class CEBlock(nn.Module):
    def __init__(self,inchs,inch):
        super(CEBlock, self).__init__()
        #self.dilate0 = nn.ConvTranspose2d(inchs, inch, kernel_size=4, stride=2, padding=1)
        self.dilate1 = nn.Conv2d(inchs, inch, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(inch, inch, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(inch, inch, kernel_size=3, dilation=4, padding=4)
        self.dilate8 = nn.Conv2d(inch, inch, kernel_size=3, dilation=8, padding=8)
        self.dilate16 = nn.Conv2d(inch, inch, kernel_size=3, dilation=16, padding=16)
        self.norm = nn.InstanceNorm2d(inchs)
        self.task0_1 = nn.Conv2d(inch, inchs, kernel_size=3, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate4(dilate2_out))
        dilate4_out = nonlinearity(self.dilate8(dilate3_out))
        dilate5_out = nonlinearity(self.dilate16(dilate4_out))

        latent = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        out0_1 = nonlinearity(self.norm(self.task0_1(latent)))
        return out0_1

from torch.nn import functional as F
class CleanU_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=70):
        super(CleanU_Net, self).__init__()
        self.filter = num_filters
       # print(self.filter)
        self.first = Conv_down2(3, self.filter, True)
        self.Conv_down1 = Conv_down(self.filter, self.filter, True)
        self.Conv_down2 = Conv_down(self.filter * 2, self.filter * 2, True)
        self.Conv_down3 = Conv_down(self.filter * 4, self.filter * 4, True)
        self.Conv_down5 = Conv_down(self.filter * 8, self.filter * 8, False)

        self.Conv_up2 = Conv_up(self.filter * 8, self.filter * 4)
        self.Conv_up1_3 = Conv_up(self.filter * 4, self.filter * 2)
        self.Conv_up2_1 = Conv_up(self.filter * 2, self.filter)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out1 = nn.Conv2d(self.filter, out_channels, 1, padding=0, stride=1)
        self.CE=CEBlock(self.filter * 8,self.filter * 8)
    def forward(self, x):
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        _, x = self.Conv_down5(x)
        x=self.CE(x)
        x = self.Conv_up2(x, conv3)
        x = self.Conv_up1_3(x, conv2)
        x2_1 = self.Conv_up2_1(x, conv1)
        x2_1 = self.up1(x2_1)
        x2_1 = self.Conv_out1(x2_1)
        x2_1=F.log_softmax(x2_1,dim=1)
        return x2_1





