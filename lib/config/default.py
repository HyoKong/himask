# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : Multi-Modal_Face_Anti-spoofing
# @Time     : 10/2/21 3:06 PM
# @File     : default.py
# @Function :

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN

_C = CN()

_C.HVD = CN()
_C.HVD.FP16_ALLREDUCE = False
_C.HVD.USE_ADASUM = False

_C.DATA = CN()
_C.DATA.DATABASE = ['HiMask']
_C.DATA.ROOT = '../database'
_C.DATA.IMG_SIZE = 144
_C.DATA.CROP_SIZE = 128
_C.DATA.LABEL_SIZE = 16
_C.DATA.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.DATA.PIXEL_STD = [0.5, 0.5, 0.5]

_C.MODEL = CN()
_C.MODEL.FFT = False
_C.MODEL.HALF_FACE = False
_C.MODEL.SMALL = False
_C.MODEL.NAME = 'CDCN'  # CDCN, CDCN_Small, HRNet_Seg, HRNet_Seg_OCR, HRNet_CLS, ViT
_C.MODEL.HRNET = '' # hrnet_w18, hrnet_w32, hrnet_w48

_C.TRAIN = CN()
_C.TRAIN.SCALE_LIST = [0, 0.4, 0.8, -0.4, -0.8]
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.WORKERS = 24
_C.TRAIN.MILESTONES = [24]
_C.TRAIN.EPOCH = 36
_C.TRAIN.GAMMA = 0.1
_C.TRAIN.LR = 1e-3
_C.TRAIN.LOSS = 'cdc'   # cls, cdc, focal, ohem
_C.TRAIN.DEBUG = 1
_C.TRAIN.SCHEDULER = 'multiStep' # multiStep, Cosine
_C.TRAIN.SOFT_LABEL = False

_C.TRAIN.LAMBDA = 0.75

_C.TRAIN.NUM_PATCHES = [1]

_C.TRAIN.RESUME = False

_C.OUTPUT = CN()
_C.OUTPUT.LOG = '../output/log/'
_C.OUTPUT.MODEL_DUMP = '../output/model_dump/'
_C.OUTPUT.WRITER = '../output/writer/'

_C.MISC = CN()
_C.MISC.DISP_FREQ = 50


def updateConfig(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    print("Loading params from cfg file...")
    cfg.freeze()
