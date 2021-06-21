# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 18/5/21 2:15 PM
# @File     : himask.py
# @Function :
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from core.transform import jigsawImg


class HiMask_Dataset(Dataset):
    def __init__(self, cfg, transforms):
        self.cfg = cfg
        self.transforms = transforms
        self.root = self.cfg.DATA.ROOT
        self.scale_list = cfg.TRAIN.SCALE_LIST
        self.imgSize = cfg.DATA.IMG_SIZE
        self.cropSize = cfg.DATA.CROP_SIZE

        with open(os.path.join(self.root, 'himask.json'), 'r') as f:
            dataDict = json.load(f)
            f.close()

        dictList = dataDict['image']
        self.db = list(dictList)

    def multi_scale(self, frame, parsing_map, scale_list, bbox, height, width):
        img_list = []
        parsing_list = []
        if random.random() < 0.1:
            scale_list = np.array(scale_list) / max(scale_list) / 20.
        for scale in scale_list:
            bbox = [int(i) for i in bbox]
            ymin, xmin, ymax, xmax = bbox
            w = xmax - xmin
            h = ymax - ymin
            y_c = int((ymin + ymax) / 2)
            x_c = int((xmin + xmax) / 2)
            ymin = max(0, int(y_c - h / 2 * (1 + scale + random.uniform(-0.05, 0.05))))
            xmin = max(0, int(x_c - w / 2 * (1 + scale + random.uniform(-0.05, 0.05))))
            ymax = min(int(y_c + h / 2 * (1 + scale + random.uniform(-0.05, 0.05))), height)
            xmax = min(int(x_c + w / 2 * (1 + scale + random.uniform(-0.05, 0.05))), width)
            img_list.append(frame[ymin:ymax, xmin:xmax, :])
            parsing_list.append(parsing_map[ymin:ymax, xmin:xmax])
        return img_list, parsing_list

    def __getitem__(self, idx):
        dataDict = self.db[idx]
        imgPath = os.path.join(self.root, dataDict['image_path'])
        parsingPath = os.path.join(self.root, dataDict['parsing_path'])
        label = int(dataDict['label'])
        bbox = dataDict['bbox']
        lmk = dataDict['landmark']
        height = dataDict['height']
        width = dataDict['width']
        ymin, xmin, ymax, xmax = bbox

        oriImg = cv2.imread(os.path.join(imgPath))
        # print(os.path.join(imgPath))
        if label:
            oriParsingImg = cv2.imread(os.path.join(parsingPath), 0)
            # oriParsingImg[oriParsingImg >0] = 1
            # oriParsingImg[oriParsingImg <=0] = -1
        else:
            oriParsingImg = np.zeros_like(oriImg[:, :, 0])
            # oriParsingImg[oriParsingImg <= 0] = -1

        imgList, parsingList = self.multi_scale(oriImg, oriParsingImg, self.scale_list, bbox, height, width)
        parsingImgList = []

        for i, (img, parsingImg) in enumerate(zip(imgList, parsingList)):
            h_r, w_r, _ = img.shape
            img = cv2.resize(img, (self.imgSize, self.imgSize))
            parsingImg = cv2.resize(parsingImg, (self.imgSize, self.imgSize))

            # random crop
            r_w = random.randint(0, self.imgSize - self.cropSize)
            r_h = random.randint(0, self.imgSize - self.cropSize)
            img = img[r_w:r_w + self.cropSize, r_h:r_h + self.cropSize, :]
            parsingImg = parsingImg[r_w:r_w + self.cropSize, r_h:r_h + self.cropSize]

            parsingImg = cv2.resize(parsingImg, (int(self.cfg.DATA.LABEL_SIZE), int(self.cfg.DATA.LABEL_SIZE))).astype(np.float32)
            parsingImg[parsingImg >= 1] = 1
            parsingImg[parsingImg <= 0] = 0

            # rotate
            angle = random.randint(-45, 45)
            cv2.rotate(img, angle)
            cv2.rotate(parsingImg, angle)

            if random.random() < 0.1:
                degree = random.randint(1, 10)
                angle = random.randint(1, 90)
                img = self.motion_blur(img, degree, angle)

            if random.random() < 0.1:
                ksize = random.randint(1, 4) * 2 - 1
                img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0, sigmaY=0)

            if self.transforms:
                # img = self.transforms(img)
                img, parsingImg = jigsawImg(img, parsingImg, numPatches=self.cfg.TRAIN.NUM_PATCHES, transforms=self.transforms)

            imgList[i] = img
            parsingImgList.append(parsingImg[np.newaxis, ...])

        data = {}
        data['img'] = imgList
        data['parsing'] = np.concatenate(parsingImgList, axis=0)
        data['label'] = label
        return data

    def __len__(self):
        return len(self.db)

    def motion_blur(self, image, degree=8, angle=45):
        image = np.array(image)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

class Evaluation_Dataset(Dataset):
    def __init__(self, cfg, transforms):
        self.cfg = cfg
        self.transforms = transforms
        self.root = self.cfg.DATA.ROOT
        self.scale_list = cfg.TRAIN.SCALE_LIST
        self.imgSize = cfg.DATA.IMG_SIZE
        self.cropSize = cfg.DATA.CROP_SIZE

        with open(os.path.join(self.root, 'himask_test.json'), 'r') as f:
            dataDict = json.load(f)
            f.close()

        dictList = dataDict['image']
        self.db = list(dictList)

    def multi_scale(self, frame, parsing_map, scale_list, bbox, height, width):
        img_list = []
        parsing_list = []
        if random.random() < 0.1:
            scale_list = np.array(scale_list) / max(scale_list) / 20.
        for scale in scale_list:
            bbox = [int(i) for i in bbox]
            ymin, xmin, ymax, xmax = bbox
            w = xmax - xmin
            h = ymax - ymin
            y_c = int((ymin + ymax) / 2)
            x_c = int((xmin + xmax) / 2)
            ymin = max(0, int(y_c - h / 2 * (1 + scale + random.uniform(-0.05, 0.05))))
            xmin = max(0, int(x_c - w / 2 * (1 + scale + random.uniform(-0.05, 0.05))))
            ymax = min(int(y_c + h / 2 * (1 + scale + random.uniform(-0.05, 0.05))), height)
            xmax = min(int(x_c + w / 2 * (1 + scale + random.uniform(-0.05, 0.05))), width)
            img_list.append(frame[ymin:ymax, xmin:xmax, :])
            parsing_list.append(parsing_map[ymin:ymax, xmin:xmax])
        return img_list, parsing_list

    def __getitem__(self, idx):
        dataDict = self.db[idx]
        imgPath = os.path.join(self.root, dataDict['image_path'])
        bbox = dataDict['bbox']
        height = dataDict['height']
        width = dataDict['width']
        ymin, xmin, ymax, xmax = bbox

        oriImg = cv2.imread(os.path.join(imgPath))
        # print(os.path.join(imgPath))

        imgList, parsingList = self.multi_scale(oriImg, np.zeros_like(oriImg[..., 0]), self.scale_list, bbox, height, width)

        for i, (img, parsingImg) in enumerate(zip(imgList, parsingList)):
            h_r, w_r, _ = img.shape
            img = cv2.resize(img, (self.imgSize, self.imgSize))

            if self.transforms:
                img = self.transforms(img)
                # img, parsingImg = jigsawImg(img, parsingImg, numPatches=self.cfg.TRAIN.NUM_PATCHES, transforms=self.transforms)

            imgList[i] = img

        data = {}
        data['img'] = imgList
        data['img_path'] = dataDict['image_path']
        return data

    def __len__(self):
        return len(self.db)
