# -*- coding: utf-8 -*-
# @Author   : Hyo Kong
# @Email    : hyokong@stu.xjtu.edu.cn
# @Project  : HiMask
# @Time     : 14/5/21 4:51 PM
# @File     : main.py
# @Function :
import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import _initPath
import random
import time
import numpy as np
import horovod.torch as hvd
import torch
import torchvision
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils.utils as utils
from config.default import _C as cfg
from config.default import updateConfig
from core.loss import Contrast_depth_loss, OhemCELoss, SoftmaxFocalLoss
from core.engine import Trainer
from head.metrics import ArcFace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiments configuration', type=str)
    args = parser.parse_args()
    updateConfig(cfg, args)
    return args


def jigsaw(imgList, parsing):
    '''

    :param imgList: [bx3xHxW, ...], len=L
    :param parsing: bxLxHxW
    :return: imgList, parsing
    '''
    B, C, H, W = imgList[0].shape
    for i, imgBatch in enumerate(imgList):
        x_h, x_w = random.randint(1, H - 1), random.randint(1, W - 1)
        r_h, r_w = random.randint(1, H - x_h), random.randint(1, W - x_w)
        selectedArea = imgBatch[:, :, x_h:(x_h + r_h), x_w:(x_w + r_w)]  # b x 3 x r_h x r_w
        shuffleIdx = [_ for _ in range(B)]
        random.shuffle(shuffleIdx)
        selectedArea = selectedArea[shuffleIdx, ...]
        imgBatch[:, :, x_h:(x_h + r_h), x_w:(x_w + r_w)] = selectedArea
        imgList[i] = imgBatch
        selectedParsing = parsing[:, i, :, :][:, int(x_h / 8):int((x_h + r_h) / 8), int(x_w / 8):int((x_w + r_w) / 8)]
        selectedParsing = selectedParsing[shuffleIdx, ...]
        parsing[:, i, :, :][:, int(x_h / 8):int((x_h + r_h) / 8), int(x_w / 8):int((x_w + r_w) / 8)] = selectedParsing
    return imgList, parsing


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = parse_args()

    # ***********************************   HOROVOD ******************************
    # init horovod
    hvd.init()
    if torch.cuda.is_available():
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(888)
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(cfg.TRAIN.WORKERS)

    kwargs = {'num_workers': cfg.TRAIN.WORKERS, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # ****************************************  SummaryWriter and wandb ******************************
    if hvd.rank() == 0:
        writer = SummaryWriter(logdir=os.path.join(cfg.OUTPUT.WRITER, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
        # wandb.init(project='multi_modal_face_antispoofing', config=cfg)

    # ************************************* TRAINING ENGINE    ****************************************
    trainer = Trainer(cfg, args, mp=hvd.size(), kwargs=kwargs)
    trainer.logger.info(' '.join(cfg))
    start_epoch = trainer.start_epoch

    if cfg.TRAIN.LOSS == 'ohem':
        criterion = OhemCELoss(thresh=0.7, n_min=14 * 14, ignore_lb=-100).cuda()
    elif cfg.TRAIN.LOSS == 'focal':
        criterion = SoftmaxFocalLoss(0.5)
    elif cfg.TRAIN.LOSS == 'cls':
        ce_criterion = nn.CrossEntropyLoss().cuda()
        # arcface = ArcFace(in_features=1024, out_features=2)
    elif cfg.TRAIN.LOSS == 'cdc':
        abs_criterion = nn.MSELoss().cuda()
        contrastive_criterion = Contrast_depth_loss().cuda()
        ce_criterion = nn.CrossEntropyLoss().cuda()

    min_loss = 1e8
    trans = torchvision.transforms.Resize(cfg.DATA.CROP_SIZE)
    parsingTrans = torchvision.transforms.Resize(cfg.DATA.LABEL_SIZE)

    # ***********************************   TRAIN   ************************************************
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        trainer.data_timer.tic()
        trainer.fp_timer.tic()
        trainer.bp_timer.tic()

        abs_losses = utils.AverageMeter()
        cont_losses = utils.AverageMeter()
        cls_losses = utils.AverageMeter()

        trainer.batch_sampler.set_epoch(epoch)

        for itr, data in enumerate(tqdm(trainer.dataloader, disable=not verbose)):
            out_mask = None
            outDict = None
            trainer.scheduler.step(epoch)
            trainer.warmup_scheduler.dampen()

            imgList = data['img']
            imgList = [i.cuda(non_blocking=True) for i in imgList]
            # for i, img in enumerate(imgList):
            # #     imgList[i] = trans(img)
            #     imgList[i] = torch.nn.functional.interpolate(img, size=[cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE])
            mask = data['parsing'].cuda(non_blocking=True)
            mask[mask == 0] = -1
            if cfg.TRAIN.SOFT_LABEL:
                mask = mask + torch.from_numpy(
                    np.random.uniform(-0.05, 0.05, size=(cfg.TRAIN.BATCH_SIZE, 1, cfg.DATA.LABEL_SIZE, cfg.DATA.LABEL_SIZE)).astype('float32')).cuda(
                    non_blocking=True)
                # mask = mask + 0.01 * torch.randn(size=(cfg.TRAIN.BATCH_SIZE, 1, cfg.DATA.LABEL_SIZE, cfg.DATA.LABEL_SIZE)).cuda(non_blocking=True)
            # mask = torch.nn.functional.interpolate(mask, size=[cfg.DATA.CROP_SIZE / 8, cfg.DATA.CROP_SIZE / 8])
            label = data['label'].cuda(non_blocking=True)
            if random.random() > 100:
                for _ in range(1):
                    imgList, mask = jigsaw(imgList, mask)
            mask_label = torch.tensor(mask, dtype=torch.long)

            trainer.data_timer.toc()
            trainer.fp_timer.tic()
            if 'HRNet' in cfg.MODEL.NAME:
                if cfg.TRAIN.LOSS not in ['cls']:
                    out_mask = trainer.model(imgList)
                else:
                    cls, out_mask = trainer.model(imgList)
            elif 'ViT' in cfg.MODEL.NAME:
                cls = trainer.model(imgList[0])
                out_mask = torch.randn((8, 3, 112, 112))
            else:
                out_mask, outDict, out_cls = trainer.model(imgList)

            if cfg.TRAIN.LOSS in ['ohem', 'ce', 'focal']:
                loss = criterion(out_mask, mask_label)
                out_mask = F.softmax(out_mask, dim=1)[:, 1, :, :]
            elif cfg.TRAIN.LOSS in ['cls']:
                loss = ce_criterion(cls, label)
                abs_losses.update(loss.data.item(), imgList[0].size(0))
                cont_losses.update(loss.data.item(), imgList[0].size(0))
            elif cfg.TRAIN.LOSS in ['cdc']:
                if not isinstance(out_mask, list):
                    out_mask = F.tanh(out_mask)
                    abs_loss = abs_criterion(out_mask, mask)
                    cont_loss = contrastive_criterion(out_mask, mask)
                    cls_loss = ce_criterion(out_cls, label)
                    loss = cfg.TRAIN.LAMBDA * abs_loss + (1 - cfg.TRAIN.LAMBDA) * cont_loss + cls_loss * 0.8
                else:
                    abs_loss = None
                    cont_loss = None
                    for _ in out_mask:
                        if abs_loss is None:
                            out_mask[0] = F.sigmoid(out_mask[0])
                            abs_loss = abs_criterion(out_mask[0], mask)
                            cont_loss = contrastive_criterion(out_mask[0], mask)
                            loss = cfg.TRAIN.LAMBDA * abs_loss + (1 - cfg.TRAIN.LAMBDA) * cont_loss
                        else:
                            out_mask[1] = F.sigmoid(out_mask[1])
                            abs_loss += abs_criterion(out_mask[1], mask)
                            cont_loss += contrastive_criterion(out_mask[1], mask)
                            loss += cfg.TRAIN.LAMBDA * abs_loss + (1 - cfg.TRAIN.LAMBDA) * cont_loss
                abs_losses.update(abs_loss.data.item(), imgList[0].size(0))
                cont_losses.update(cont_loss.data.item(), imgList[0].size(0))
                cls_losses.update(cls_loss.data.item(), imgList[0].size(0))

            trainer.fp_timer.toc()
            trainer.bp_timer.tic()

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            trainer.bp_timer.toc()
            trainer.misc_timer.tic()

            screen = [
                'Epoch {}/{} Itr {}/{} | '.format(epoch, cfg.TRAIN.EPOCH, itr, trainer.itr_per_epoch),
                'lr {:6f} | '.format(trainer.get_lr()),
                'abs loss {:.6f} cont loss {:.6f} cls loss {:.6f} | '.format(abs_losses.avg, cont_losses.avg, cls_losses.avg),
                'data time {:.3f} fp time {:.3f} bp time {:.3f} misc time: {:.3f}'.format(trainer.data_timer.average_time, trainer.fp_timer.average_time,
                                                                                          trainer.bp_timer.average_time, trainer.misc_timer.average_time)
            ]

            if itr % cfg.MISC.DISP_FREQ == 0 and verbose:
                trainer.logger.info(' '.join(screen))

            trainer.misc_timer.toc()
            trainer.data_timer.tic()

        if verbose:
            if abs_losses.avg * cfg.TRAIN.LAMBDA + cont_losses.avg * (1 - cfg.TRAIN.LAMBDA) + cls_loss * 0.5 < min_loss:
                min_loss = abs_losses.avg * cfg.TRAIN.LAMBDA + cont_losses.avg * (1 - cfg.TRAIN.LAMBDA) + cls_loss * 0.5
                trainer.save_model({
                    'epoch'    : epoch,
                    'network'  : trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                }, 0)
            file_path = os.path.join(cfg.OUTPUT.MODEL_DUMP, '{}_snapshot_{}.h5'.format(os.path.basename(args.cfg)[:-5], str(epoch)))
            # wandb.save(file_path)

            writer.add_scalar('{}/loss/abs_loss'.format(os.path.basename(args.cfg).split('.')[0]), abs_losses.avg, epoch)
            writer.add_scalar('{}/loss/cont_loss'.format(os.path.basename(args.cfg).split('.')[0]), cont_losses.avg, epoch)
            writer.add_scalar('{}/loss/cls_loss'.format(os.path.basename(args.cfg).split('.')[0]), cls_losses.avg, epoch)
            # wandb.log({'loss/cont_loss'.format(): cont_losses.avg, 'loss/abs_loss'.format(): abs_losses.avg})
            # wandb.log({'loss/abs_loss'.format(): abs_losses.avg})
            if outDict is not None:
                for k, v in outDict.items():
                    v.cuda()
                    if 'x' not in k:
                        v = torch.sum(v, dim=1, keepdim=True)
                    if cfg.MODEL.FFT and 'x' not in k:
                        x = vutils.make_grid(v.data[:, :3, :, :], normalize=True, scale_each=True)
                        writer.add_image('{}/rgb/{}'.format(os.path.basename(args.cfg).split('.')[0], k), x, epoch)
                        x = vutils.make_grid(torch.cat((v.data[:, 3, :, :].unsqueeze(1), v.data[:, 3, :, :].unsqueeze(1), v.data[:, 3, :, :].unsqueeze(1)), 1),
                                             normalize=True, scale_each=True)
                        writer.add_image('{}/rgb_fft1/{}'.format(os.path.basename(args.cfg).split('.')[0], k), x, epoch)
                        x = vutils.make_grid(torch.cat((v.data[:, 4, :, :].unsqueeze(1), v.data[:, 4, :, :].unsqueeze(1), v.data[:, 4, :, :].unsqueeze(1)), 1),
                                             normalize=True, scale_each=True)
                        writer.add_image('{}/rgb_fft2/{}'.format(os.path.basename(args.cfg).split('.')[0], k), x, epoch)
                        x = vutils.make_grid(torch.cat((v.data[:, 5, :, :].unsqueeze(1), v.data[:, 5, :, :].unsqueeze(1), v.data[:, 5, :, :].unsqueeze(1)), 1),
                                             normalize=True, scale_each=True)
                        writer.add_image('{}/rgb_fft3/{}'.format(os.path.basename(args.cfg).split('.')[0], k), x, epoch)
                    else:
                        x = vutils.make_grid(torch.flip(v.data[:, :3, :, :], dims=[1]), normalize=True, scale_each=True)
                        writer.add_image('{}/rgb/{}'.format(os.path.basename(args.cfg).split('.')[0], k), x, epoch)
            if isinstance(out_mask, list):
                for each in out_mask:
                    x = vutils.make_grid(each.data[:, 0, :, :].unsqueeze(1), normalize=True, scale_each=True)
                    writer.add_image('{}/mask'.format(os.path.basename(args.cfg).split('.')[0]), x, epoch)
            else:
                x = vutils.make_grid(out_mask.data[:, 0, :, :].unsqueeze(1), normalize=True, scale_each=True)
                writer.add_image('{}/mask'.format(os.path.basename(args.cfg).split('.')[0]), x, epoch)
            x = vutils.make_grid(torch.flip(imgList[0].data[:, :, :, :], dims=(1,)), normalize=True, scale_each=True)
            writer.add_image('{}/img'.format(os.path.basename(args.cfg).split('.')[0]), x, epoch)

            # wandb.log({'{}/mask'.format(os.path.basename(args.cfg).split('.')[0]): [wandb.Image(i) for i in out_mask]})
            # mask = mask.unsqueeze(1)
            # temp = mask.data[:, 0, :, :]
            # temp[temp==-1] = 0
            x = vutils.make_grid(mask.data[:, 0, :, :].unsqueeze(1), normalize=True, scale_each=True)
            writer.add_image('{}/mask_gt'.format(os.path.basename(args.cfg).split('.')[0]), x, epoch)
            # wandb.log({'{}/mask_gt'.format(os.path.basename(args.cfg).split('.')[0]): [wandb.Image(i) for i in mask]})

    if verbose:
        # wandb.run.finish() if wandb and wandb.run else None
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
