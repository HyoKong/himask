# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : Multi-Modal_Face_Anti-spoofing
# @Time     : 17/2/21 4:56 PM
# @File     : loss.py
# @Function :

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import horovod.torch as hvd


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)  # 8x1x3x3
    if len(input.shape) == 3:
        input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])    # bx8xhxw
        contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    elif len(input.shape) == 4:
        # if hvd.rank() == 0:
        #     import ipdb
        #     ipdb.set_trace()
        kernel_filter = kernel_filter.repeat(input.shape[1], 1, 1, 1)
        input = input.repeat(1, 8, 1, 1)

        contrast_depth = F.conv2d(input, weight=kernel_filter, groups=kernel_filter.shape[0])  # depthwise conv

    return contrast_depth


class Contrast_depth_loss(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label)
        # loss = torch.pow(contrast_out - contrast_label, 2)
        # loss = torch.mean(loss)

        return loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
        # self.criteria = nn.NLLLoss()

    def forward(self, logits, labels):
        # N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        # log_score = torch.log(logits)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss
