# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 20/5/21 4:49 PM
# @File     : eval.py
# @Function :

import argparse
import json
import os
import _initPath
import cv2
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from detector_pytorch import Detector_Pytorch

from config.default import _C as cfg
from config.default import updateConfig
from model.CDCN import CDCN_Multi, CDCN_Small
from model.seg_hrnet import HighResolutionNet as HighResolutionNet_Seg
from model.seg_hrnet_ocr import HighResolutionNet as HighResolutionNet_Ocr
from model.cls_hrnet import HighResolutionNet as HighResolutionNet_Cls
from model.vit import ViT as ViT

from data.himask import Evaluation_Dataset
from data.MPDataLoader import DataLoader
from collections import OrderedDict

import zipfile
import xlsxwriter
from utils.export2excel import ResultWriter


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_model(modelPath, model):
    ckpt = torch.load(modelPath, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        name = 'module.' + k
        new_state_dict[name] = v
    del ckpt
    model.load_state_dict(new_state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configuration', type=str)
    parser.add_argument('--epoch', help='epoch of the loaded model', type=int)
    parser.add_argument('--gpu', help='gpu')
    args = parser.parse_args()
    updateConfig(cfg, args)
    return args


def multi_scale(frame, parsing_map, scale_list, bbox, height, width):
    img_list = []
    parsing_list = []
    # if random.random() < 0.2:
    #     scale_list = np.array(scale_list) / max(scale_list) / 100.
    for scale in scale_list:
        bbox = [int(i) for i in bbox]
        ymin, xmin, ymax, xmax = bbox
        w = xmax - xmin
        h = ymax - ymin
        y_c = int((ymin + ymax) / 2)
        x_c = int((xmin + xmax) / 2)
        ymin = max(0, int(y_c - h / 2 * (1 + scale)))
        xmin = max(0, int(x_c - w / 2 * (1 + scale)))
        ymax = min(int(y_c + h / 2 * (1 + scale)), height)
        xmax = min(int(x_c + w / 2 * (1 + scale)), width)
        img_list.append(frame[ymin:ymax, xmin:xmax, :])
        parsing_list.append(parsing_map[ymin:ymax, xmin:xmax])
    return img_list, parsing_list


def read_txt(txtPath):
    with open(txtPath, 'r') as f:
        lines = f.readlines()
    f.close()
    for i, line in enumerate(lines):
        lines[i] = line.strip()
    return lines

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # init path
    imgRoot = '../database/'
    txtPath = '../database/test.txt'
    resultTxtPath = '../database/checked_val.txt'
    # gtDict = {}
    # with open(resultTxtPath, 'r') as f:
    #     lines = f.readlines()
    #     f.close()
    # for line in lines:
    #     line = line.strip().split(' ')
    #     gtDict[line[0]] = line[1]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATA.PIXEL_MEAN, std=cfg.DATA.PIXEL_STD)
    ])
    dataset = Evaluation_Dataset(cfg, transform)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=80, shuffle=False, num_workers=24, drop_last=False)

    # init model
    modelPath = os.path.join(cfg.OUTPUT.MODEL_DUMP, '{}_snapshot_{}.h5'.format(os.path.basename(args.cfg)[:-5], str(args.epoch)))
    modelDict = {}
    modelDict['CDCN'] = CDCN_Multi(cfg=cfg, fft=cfg.MODEL.FFT, feats=False)
    modelDict['CDCN_Small'] = CDCN_Small(cfg=cfg, fft=cfg.MODEL.FFT)
    modelDict['HRNet_SEG'] = HighResolutionNet_Seg(cfg)
    modelDict['HRNet_SEG_OCR'] = HighResolutionNet_Ocr(cfg)
    modelDict['HRNet_CLS'] = HighResolutionNet_Cls(cfg)
    modelDict['ViT'] = ViT(image_size=256, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
    model = modelDict[cfg.MODEL.NAME]

    # faceDetector = Detector_Pytorch()

    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model = load_model(modelPath, model)
    model.cuda()

    model.eval()
    device = torch.device("cuda:0")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.DATA.IMG_SIZE),
        transforms.CenterCrop(cfg.DATA.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # imgPathList = read_txt(txtPath)
    resultDict = {}
    resultDict1 = {}
    resultDict2 = {}
    count = 1
    # for i, imgPath in enumerate(tqdm(imgPathList)):
    for i, dataDict in enumerate(tqdm(dataLoader)):

        imgList = dataDict['img']
        imgPathList = dataDict['img_path']
        imgList = [i.cuda(non_blocking=True) for i in imgList]

        with torch.no_grad():
            if 'HRNet' in cfg.MODEL.NAME:
                if cfg.TRAIN.LOSS not in ['cls']:
                    out_mask = model(imgList)
                else:
                    cls, out_mask = model(imgList)
                    cls = cls[:,1]

                if not isinstance(out_mask, list):
                    out_mask = F.sigmoid(out_mask)
                else:
                    for _, each in enumerate(out_mask):
                        out_mask[_] = F.sigmoid(each)
                    out_mask = torch.cat((out_mask[0], out_mask[1]), dim=1)
            elif 'ViT' in cfg.MODEL.NAME:
                cls = model(imgList[0])
                out_mask = torch.randn((8, 3, 16, 16))
                cls = cls[0][1]
            else:
                # out_mask, outDict = model(imgList)
                out_mask, cls = model(imgList)
                # cls = F.sigmoid(cls)
                out_mask = F.sigmoid(out_mask * 1000)
        out_mask = torch.sum(out_mask, dim=1).data.cpu().numpy()
        # mask = out_mask[0, ...].data.cpu().numpy()
        score_list = []
        score_list1  = []
        score_list2  = []
        for idx, each_mask in enumerate(out_mask):
            if cfg.TRAIN.LOSS not in ['cls']:
                # score = np.mean(each_mask.data.cpu().numpy())
                score_list.append(np.mean(each_mask) * sigmoid(6.66 * cls[idx][1].cpu()))
                score_list1.append(np.mean(each_mask))
                score_list2.append(sigmoid(6.66 * cls[idx][1].cpu()))
            else:
                score = cls
                score_list += score.cpu().numpy().tolist()
            # score_list.append(score)
            # score_list += score.cpu().numpy().tolist()
        for each_score, imgPath in zip(score_list, imgPathList):
            resultDict[imgPath] = each_score
        for each_score, imgPath in zip(score_list1, imgPathList):
            resultDict1[imgPath] = each_score
        for each_score, imgPath in zip(score_list2, imgPathList):
            resultDict2[imgPath] = each_score
        # if int(gtDict[imgPath]) == 0 and score <=0:
        #     excelWriter.writeRow(oriImg, imgPath, score)
        # elif int(gtDict[imgPath]) == 1 and score >=0:
        #     excelWriter.writeRow(oriImg, imgPath, score)

    # excelWriter.close()
    # save results
    resultPath = os.path.join('..', 'result', '{}.txt'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    if not os.path.exists(os.path.dirname(resultPath)):
        os.makedirs(os.path.dirname(resultPath), exist_ok=True)
    count = 1
    with open(resultPath, 'w') as f:
        for k, v in resultDict2.items():
            f.write('{} {}\n'.format(k, v))
        f.close()

    # with open(os.path.join(os.path.dirname(resultPath), 'parsing.txt'), 'w') as f:
    #     for k, v in resultDict1.items():
    #         f.write('{} {}\n'.format(k, v))
    #     f.close()
    #
    # with open(os.path.join(os.path.dirname(resultPath), 'cls.txt'), 'w') as f:
    #     for k, v in resultDict2.items():
    #         f.write('{} {}\n'.format(k, v))
    #     f.close()
    # f = zipfile.ZipFile(resultPath, 'w', zipfile.ZIP_DEFLATED)
    # f.write(resultPath.replace('txt', 'zip'))
    # f.close()
