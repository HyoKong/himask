# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 18/5/21 1:21 PM
# @File     : engine.py
# @Function :

import math
import os

import horovod.torch as hvd
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.multiple_datasets import MultipleDatasets
from model.CDCN import CDCN_Multi, CDCN_Small
from model.seg_hrnet import HighResolutionNet as HighResolutionNet_Seg
from model.seg_hrnet_ocr import HighResolutionNet as HighResolutionNet_Ocr
from model.cls_hrnet import HighResolutionNet as HighResolutionNet_Cls
from model.vit import ViT as ViT
from utils import utils
from utils.logger import colorlogger
from core.warmup import BaseWarmup, LinearWarmup, ExponentialWarmup
from core.scheduler import WarmupConstantSchedule, WarmupCosineSchedule


class Trainer(object):
    def __init__(self, cfg, args, mp, kwargs, train=True):
        self.cfg = cfg
        self.args = args
        self.mp = mp
        self.kwargs = kwargs
        self.log_name = os.path.basename(self.args.cfg).split('.')[0] + "_train.txt"

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.8, contrast=0.15, hue=0.15, saturation=0.15),
            # transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(value=0),  # value='random'
            transforms.Normalize(mean=self.cfg.DATA.PIXEL_MEAN, std=self.cfg.DATA.PIXEL_STD),
        ])

        modelDict = {}
        modelDict['CDCN'] = CDCN_Multi(cfg=cfg, fft=cfg.MODEL.FFT)
        modelDict['CDCN_Small'] = CDCN_Small(cfg=cfg, fft=cfg.MODEL.FFT)
        modelDict['HRNet_SEG'] = HighResolutionNet_Seg(cfg)
        modelDict['HRNet_SEG_OCR'] = HighResolutionNet_Ocr(cfg)
        modelDict['HRNet_CLS'] = HighResolutionNet_Cls(cfg)
        modelDict['ViT'] = ViT(image_size=256, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048,dropout=0.1, emb_dropout=0.1)
        model = modelDict[self.cfg.MODEL.NAME]

        if self.cfg.TRAIN.RESUME:
            modelPath = os.path.join(cfg.OUTPUT.MODEL_DUMP, '{}_snapshot_{}.h5'.format(os.path.basename(args.cfg)[:-5], str(0)))
            model, self.start_epoch = self.load_model(modelPath, model)
        else:
            self.start_epoch = 0

        model.cuda()

        self.model = self.build_model(model)
        if train:
            self.model.train()
        else:
            self.model.eval()
        self.dataloader = self.build_dataloader()
        self.optimizer, self.scheduler = self.build_optim(model)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=300)
        self.warmup_scheduler.last_step = -1
        del model

        self.logger = colorlogger(self.cfg.OUTPUT.LOG, log_name=self.log_name)

        self.data_timer = utils.Timer()
        self.fp_timer = utils.Timer()
        self.bp_timer = utils.Timer()
        self.misc_timer = utils.Timer()

        self.itr_per_epoch = math.ceil(len(self.dataloader) / hvd.size())

    def build_model(self, model):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        return model

    def build_dataloader(self):
        # dataset = Surf_Dataset(self.cfg, self.transform_rgb, self.transform_rgb)
        datasetList = []
        for data in self.cfg.DATA.DATABASE:
            # if hvd.rank() == 0:
            #     import ipdb
            #     ipdb.set_trace()
            exec('from data.{} import {}_Dataset as {}_Dataset'.format(data.lower(), data, data))
            trainset = eval('{}_Dataset'.format(data))
            datasetList.append(trainset(self.cfg, self.transforms))
        dataset = MultipleDatasets(datasetList)
        batch_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
        self.batch_sampler = batch_sampler
        dataloader = DataLoader(dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, sampler=batch_sampler,   # , num_workers=self.cfg.TRAIN.WORKERS, pin_memory=True,
                                drop_last=True, **self.kwargs)
        return dataloader

    def separate_resnet_bn_paras(self, modules):
        all_parameters = modules.parameters()
        paras_only_bn = []

        for pname, p in modules.named_parameters():
            if pname.find('bn') >= 0:
                paras_only_bn.append(p)

        paras_only_bn_id = list(map(id, paras_only_bn))
        paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

        return paras_only_bn, paras_wo_bn

    def build_optim(self, model):
        model_without_ddp = model
        # param_dicts = [
        #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        #     {
        #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        #         "lr"    : self.cfg.TRAIN.LR,
        #     },
        # ]
        backbone_paras_only_bn, backbone_paras_wo_bn = self.separate_resnet_bn_paras(model_without_ddp)

        if 'HRNet' in self.cfg.MODEL.NAME or 'ViT' in self.cfg.MODEL.NAME:
            param_dicts = [{'params': model_without_ddp.parameters()}]
        else:
            param_dicts = [{'params': backbone_paras_wo_bn, 'weight_decay': 1e-5}, {'params': backbone_paras_only_bn}]

        if hvd.nccl_built():
            lr_scaler = hvd.local_size()
        else:
            lr_scaler = None

        if not lr_scaler:
            if not 'ViT' in self.cfg.MODEL.NAME:
                optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.TRAIN.LR, weight_decay=1e-5)
                # optimizer = Optimizer(self.model, lr0=self.cfg.TRAIN.LR, momentum=0.9, wd=5e-4, warmup_steps=400, warmup_start_lr=1e-4,
                #                       max_iter=int(len(self.dataloader) / hvd.size()) * self.cfg.TRAIN.EPOCH, power=0.9)
            else:
                optimizer = torch.optim.SGD(param_dicts, lr=self.cfg.TRAIN.LR, momentum=0.9, weight_decay=1e-7)
        else:
            if not 'ViT' in self.cfg.MODEL.NAME:
                optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.TRAIN.LR * lr_scaler, weight_decay=1e-5)
                # optimizer = Optimizer(self.model, lr0=self.cfg.TRAIN.LR, momentum=0.9, wd=5e-4, warmup_steps=400, warmup_start_lr=1e-4,
                #                       max_iter=int(len(self.dataloader) / hvd.size()) * self.cfg.TRAIN.EPOCH, power=0.9)
            else:
                optimizer = torch.optim.SGD(param_dicts, lr=self.cfg.TRAIN.LR * lr_scaler, momentum=0.9, weight_decay=1e-7)

        if self.cfg.TRAIN.RESUME:
            modelPath = os.path.join(self.cfg.OUTPUT.MODEL_DUMP, '{}_snapshot_{}.h5'.format(os.path.basename(self.args.cfg)[:-5], str(0)))
            optimizer = self.load_optimizer(modelPath, optimizer)

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)  # optimizer.optim for SGD optimizer
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.cfg.HVD.FP16_ALLREDUCE else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer,  # optimizer.optim for SGD optimizer
                                             named_parameters=model.named_parameters(),
                                             compression=compression,
                                             op=hvd.Adasum if self.cfg.HVD.USE_ADASUM else hvd.Average,
                                             # gradient_predivide_factor=1.0,
                                             )
        # lr scheduler
        # if not 'ViT' in self.cfg.MODEL.NAME:
        #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.TRAIN.MILESTONES, gamma=self.cfg.TRAIN.GAMMA)
        # else:
        #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=int(len(self.dataloader) / hvd.size() * self.cfg.TRAIN.EPOCH))
        scheduler_dict = {}
        scheduler_dict['multiStep'] = lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.TRAIN.MILESTONES, gamma=self.cfg.TRAIN.GAMMA)
        # scheduler_dict['Cosine'] = WarmupCosineSchedule(optimizer, warmup_steps=200, t_total=int(len(self.dataloader) / hvd.size() * self.cfg.TRAIN.EPOCH))
        # scheduler_dict['Cosine'] = lr_scheduler.CosineAnnealingWarmRestarts()
        return optimizer, scheduler_dict[self.cfg.TRAIN.SCHEDULER]

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def save_model(self, state, epoch):
        file_path = os.path.join(self.cfg.OUTPUT.MODEL_DUMP, '{}_snapshot_{}.h5'.format(os.path.basename(self.args.cfg)[:-5], str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, modelPath, model):
        ckpt = torch.load(modelPath, map_location='cpu')
        state_dict = ckpt['network']
        start_epoch = ckpt['epoch']

        # new_state_dict = OrderedDict()
        # for k, v in ckpt['network'].items():
        #     name = 'module.' + k
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)
        return model, start_epoch

    def load_optimizer(self, modelPath, optimizer):
        ckpt = torch.load(modelPath, map_location='cpu')
        state_dict = ckpt['optimizer']
        optimizer.load_state_dict(state_dict)
        return optimizer