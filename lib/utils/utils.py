# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : Multi-Modal_Face_Anti-spoofing
# @Time     : 17/2/21 4:00 PM
# @File     : utils.py
# @Function :

import time
import numpy as np


def gaussian_k(x0, y0, sigma, width, height):
    """
    Make a square gaussian kernel centered at (x0,y0) with sigma
    :param x0:
    :param y0:
    :param sigma:
    :param width:
    :param height:
    :return: Square gaussian kernel
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10:
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
