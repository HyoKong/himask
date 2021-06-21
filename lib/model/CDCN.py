# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : Multi-Modal_Face_Anti-spoofing
# @Time     : 10/2/21 2:52 PM
# @File     : CDCN.py
# @Function :

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.loss import SoftmaxFocalLoss
import torchvision.transforms as transforms


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class Conv2d_CD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode='replicate')
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.) < 1e-8:
            return out_normal
        else:
            # [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation,
                                groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class CDCN_Multi(nn.Module):
    def __init__(self, cfg, basic_conv=Conv2d_CD, theta=0.7, fft=False, bias=False, feats=True):
        super().__init__()
        self.norm = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.cfg = cfg
        self.feats = feats
        self.scale_list = cfg.TRAIN.SCALE_LIST
        if fft:
            in_channel = 6 * len(self.scale_list)

        else:
            in_channel = 3 * len(self.scale_list)

        out_channel = len(self.scale_list)
        for i, scale in enumerate(self.scale_list):
            setattr(self, 'conv_m{}'.format(i + 1),
                    nn.Sequential(
                        basic_conv(3, 64, kernel_size=7, stride=2, padding=3, bias=bias, theta=theta),
                        self.norm(64),
                        self.relu,
                    )
                    )

            setattr(self, 'block1_m{}'.format(i + 1),
                    nn.Sequential(
                        basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        self.relu,
                    )
                    )

            setattr(self, 'block2_m{}'.format(i + 1),
                    nn.Sequential(
                        basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        self.relu,
                        basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(196),
                        self.relu,
                        basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        self.relu,
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )
                    )

            setattr(self, 'block3_m{}'.format(i + 1),
                    nn.Sequential(
                        basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        self.relu,
                        basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(196),
                        self.relu,
                        basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        self.relu,
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    )
                    )

            setattr(self, 'last_conv_m{}'.format(i + 1),
                    nn.Sequential(
                        basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                        self.norm(128),
                        nn.ReLU(inplace=True),
                    )
                    )

        # setattr(self, 'last_conv1',
        #         nn.Sequential(
        #             basic_conv(128 * len(self.scale_list), 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #             self.norm(128),
        #             nn.ReLU(inplace=True),
        #         )
        #         )
        self.last_conv1 = nn.Sequential(
            basic_conv(128 * len(self.scale_list), 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
            self.norm(128),
            self.relu,
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
            self.norm(196),
            self.relu,
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
            self.norm(128),
            # self.relu,
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.final_conv1x1 = nn.Conv2d(128 * len(self.scale_list), 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid_conv1x1 = nn.Conv2d(128 * 3, 128, kernel_size=1, stride=1, padding=0, bias=False)
        setattr(self, 'last_conv2',
                nn.Sequential(
                    basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                    self.norm(128),
                    self.relu,
                    # basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                    # self.norm(196),
                    # self.relu,
                    basic_conv(128, out_channel, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
                    # nn.ReLU(),
                )
                )

        # self.conv1_M1 = nn.Sequential(
        #     basic_conv(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(64),
        #     self.relu,
        # )
        #
        # self.Block1_M1 = nn.Sequential(
        #     basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     self.relu,
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        # )
        #
        # self.Block2_M1 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     self.relu,
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        #
        # self.Block3_M1 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     self.relu,
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        #
        # self.conv1_M2 = nn.Sequential(
        #     basic_conv(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(64),
        #     self.relu,
        # )
        #
        # self.Block1_M2 = nn.Sequential(
        #     basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     self.relu,
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     self.relu,
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        # )
        #
        # self.Block2_M2 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     nn.ReLU(),
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        #
        # self.Block3_M2 = nn.Sequential(
        #     basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        #     basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(196),
        #     nn.ReLU(),
        #     basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        #
        # self.lastconv1_M1 = nn.Sequential(
        #     basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        # )
        # self.lastconv1_M2 = nn.Sequential(
        #     basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        # )
        #
        # self.lastconv2 = nn.Sequential(
        #     basic_conv(128 * 2, 128, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     self.norm(128),
        #     nn.ReLU(),
        # )
        # self.lastconv3 = nn.Sequential(
        #     basic_conv(128, out_channel, kernel_size=3, stride=1, padding=1, bias=bias, theta=theta),
        #     # nn.ReLU(),
        # )
        self.down = nn.AdaptiveMaxPool2d(int(self.cfg.DATA.CROP_SIZE / 8))
        self.globalPool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(512)
        self.fc = nn.Linear(cfg.DATA.LABEL_SIZE * cfg.DATA.LABEL_SIZE * 128, 512)
        self.fc_out = nn.Linear(512, 2)

    def forward(self, imgList):
        all = None
        out = {}
        for i, img in enumerate(imgList):
            out['x_{}'.format(i + 1)] = img
            conv = getattr(self, 'conv_m{}'.format(i + 1))(img)
            out['conv_{}'.format(i + 1)] = conv

            b1 = getattr(self, 'block1_m{}'.format(i + 1))(conv)
            b1_down = self.down(b1)
            out['block1_m{}'.format(i + 1)] = b1

            b2 = getattr(self, 'block2_m{}'.format(i + 1))(b1)
            b2_down = self.down(b2)
            out['block2_m{}'.format(i + 1)] = b2

            b3 = getattr(self, 'block3_m{}'.format(i + 1))(b2)
            b3_down = self.down(b3)
            out['block3_m{}'.format(i + 1)] = b3

            b_cat = torch.cat((b1_down, b2_down, b3_down), dim=1)

            b_cat_global = self.globalPool(b_cat)

            b = getattr(self, 'last_conv_m{}'.format(i + 1))(b_cat) * self.sigmoid(self.mid_conv1x1(b_cat_global))
            out['block_final{}'.format(i + 1)] = b

            if all is None:
                all = b
            else:
                all = torch.cat((all, b), dim=1)

        all_global = self.globalPool(all)
        all = self.last_conv1(all) * self.sigmoid(self.final_conv1x1(all_global))

        cls = all.view(all.size(0), -1)
        cls = self.fc(cls)
        cls = self.bn1(cls)
        cls = self.fc_out(cls)
        # cls =

        out['feat'] = all
        map_x = self.last_conv2(all)
        if self.feats:
            return map_x, out, cls
        else:
            return map_x, cls

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class CDCN_Small(nn.Module):
    def __init__(self, cfg, fft=False):
        super().__init__()
        self.norm = nn.BatchNorm2d
        self.relu = nn.ReLU

        self.scale_list = cfg.TRAIN.SCALE_LIST
        if fft:
            in_channel = 6 * len(self.scale_list)

        else:
            in_channel = 3 * len(self.scale_list)
        out_channel = len(self.scale_list)

        self.conv_M1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 2, 1, bias=False),
            self.norm(64),
            self.relu(inplace=True)
        )
        self.conv_M2 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 2, 1, bias=False),
            self.norm(64),
            self.relu(inplace=True)
        )
        channel = 128

        self.Block1 = nn.Sequential(
            nn.Conv2d(64 * 2, channel, kernel_size=3, stride=2, padding=1, bias=False),
            self.norm(channel),
            self.relu(inplace=True),
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(128),
        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(128, channel, kernel_size=3, stride=2, padding=1, bias=False),
            self.norm(channel),
            self.relu(inplace=True),
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(128),
        )
        self.Block3 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # self.norm(64),
            # self.relu(),
            nn.Conv2d(128, channel, kernel_size=3, stride=2, padding=1, bias=False),
            self.norm(channel),
            self.relu(inplace=True),
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(128),
        )

        self.downsample14 = nn.Upsample(size=(14, 14), mode='bilinear')

        self.lastconv = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=3, stride=1, padding=1, bias=False),
            self.norm(128),
            self.relu(inplace=True),
            nn.Conv2d(128, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x1, x2):
        x1_conv = self.conv_M1(x1)

        x2_conv = self.conv_M2(x2)
        x = torch.cat((x1_conv, x2_conv), dim=1)

        x_Block1 = self.Block1(x)
        x_Block1_14 = self.downsample14(x_Block1)

        x_Block2 = self.Block2(x_Block1)

        x_cat = torch.cat((x_Block1_14, x_Block2), dim=1)
        map_x = self.lastconv(x_cat)

        rgbOut = {}
        rgbOut['input'] = x1
        rgbOut['conv1_M1'] = x1_conv
        rgbOut['feat'] = x_cat
        rgbOut['feat_blk1'] = x_Block1
        rgbOut['feat_blk2'] = x_Block2

        irOut = {}
        irOut['input'] = x2
        irOut['conv1_M2'] = x2_conv
        irOut['feat'] = x_cat
        irOut['feat_blk1'] = x_Block1
        irOut['feat_blk2'] = x_Block2

        return map_x, rgbOut, irOut

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == '__main__':
    input1 = torch.randn((8, 3, 112, 112))
    input2 = torch.randn((8, 3, 112, 112))
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    # input1 = transform(input1)
    # input2 = transform(input2)

    input1 = input1.sub_(mean).div_(std)
    input2 = input2.sub_(mean).div_(std)

    lbs = torch.randint(0, 1, [8, 14, 14])
    criteria = SoftmaxFocalLoss(0.5)
    net = CDCN_Multi()
    map_x, rgbOut, irOut = net(input1, input2)
    loss = criteria(map_x, lbs)

    for k, v in rgbOut.items():
        if torch.sum(torch.isinf(v)) > 0 or torch.sum(torch.isnan(v)) > 0:
            import ipdb

            ipdb.set_trace()

    print(loss.detach().cpu())
