# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 10/6/21 1:28 PM
# @File     : transform.py
# @Function :

import cv2
import os
import random
import numpy as np
import torchvision
import torch
import torch.nn as nn
from PIL import Image
import imgaug.augmenters as iaa


def jigsawImg(img, parsingImg, numPatches, transforms):
    imgHeight, imgWidth, imgChannel = img.shape
    parsingHeight, parsingWidth = parsingImg.shape
    # img = Image.fromarray(img)
    # parsingImg = Image.fromarray(parsingImg, 'L')
    numPatches = random.choice(numPatches)
    patchSize = int(imgHeight / numPatches)
    scale = int(imgHeight / parsingHeight)

    tiles = [None] * numPatches ** 2
    parsingTiles = [None] * numPatches ** 2
    orders = np.arange(numPatches ** 2).tolist()
    random.shuffle(orders)
    pad = torch.nn.ZeroPad2d(padding=(0,1,0,1))

    for n in range(numPatches ** 2):
        i = n // numPatches
        j = n % numPatches
        # tile = img.crop([patchSize * j, patchSize * i, (j + 1) * patchSize, (i + 1) * patchSize])
        # if (j + 1) % numPatches == 0 and (i + 1) % numPatches == 0:
        #     tile = img[int(patchSize * i):int(patchSize * (i + 1) + imgHeight % patchSize), int(patchSize * j):int(patchSize * (j + 1) + imgHeight % patchSize),
        #            :]
        # elif (j + 1) % numPatches == 0:
        #     tile = img[int(patchSize * i):int(patchSize * (i + 1)), int(patchSize * j):int(patchSize * (j + 1) + imgWidth % patchSize), :]
        # elif (i + 1) % numPatches == 0:
        #     tile = img[int(patchSize * i):int(patchSize * (i + 1) + imgHeight % patchSize), int(patchSize * j):int(patchSize * (j + 1)), :]
        # else:
        tile = img[int(patchSize * i):int(patchSize * (i + 1)), int(patchSize * j):int(patchSize * (j + 1)), :]
        tile = transforms(tile)
        tiles[n] = tile
        # parsingTile = parsingImg.crop([patchSize * j / scale, patchSize * i / scale, (j + 1) * patchSize / scale, (i + 1) * patchSize / scale])
        parsingTile = parsingImg[int(patchSize * i / scale):int(patchSize * (i + 1) / scale), int(patchSize * j / scale):int(patchSize * (j + 1) / scale)]
        parsingTiles[n] = np.array(parsingTile)

    combinedImg = None
    combinedParsing = None
    for i in range(numPatches):
        rowImg = None
        rowParsing = None
        for j in range(numPatches):
            if rowImg is None:
                rowImg = tiles[orders[i * numPatches + j]]
                rowParsing = parsingTiles[orders[i * numPatches + j]]
            else:
                rowImg = torch.cat((rowImg, tiles[orders[i * numPatches + j]]), dim=2)
                rowParsing = np.concatenate((rowParsing, parsingTiles[orders[i * numPatches + j]]), axis=1)
        if combinedImg is None:
            combinedImg = rowImg
            combinedParsing = rowParsing
        else:
            combinedImg = torch.cat((combinedImg, rowImg), dim=1)
            combinedParsing = np.concatenate((combinedParsing, rowParsing), axis=0)
    # combinedImg = torch.nn.functional.interpolate(combinedImg, size=([imgChannel, imgHeight, imgWidth]))
    # combinedParsing = np.reshape(combinedParsing, (parsingHeight, parsingWidth))
    # combinedParsing = np.interp()
    if imgHeight % numPatches!= 0:
        combinedParsing = np.pad(combinedParsing, ((0,1),(0,1)),constant_values=(0,0))
        combinedImg = pad(combinedImg)
    return combinedImg, combinedParsing


def jigsawBatchImg(batchImg, numPatches, transforms):
    return 0
