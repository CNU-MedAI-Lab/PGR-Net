# -*- coding: utf-8 -*-
# 删除了验证部分
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.backends.cudnn as cudnn
import torchvision

from run.dataset import Dataset
from utils.str2bool import str2bool
from utils import count_params, losses
from utils.metrics import dice_coef, batch_iou, mean_iou, iou_score

import pandas as pd
# import unet
# from models import unet
from utils.metrics import dice_coef
from model.pgr_net import UNetRetNet

# arch_names = list(VSSM.__dict__.keys())

loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Unet_MICCAI",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        help='model architecture: ' +
                             ' | '.join('UNet') +
                             ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)  # Br20_flair_high_Enhance_WT
    parser.add_argument('--Model', default="pgr",  # Brats20frq Br20high_Enhance  Br20_Some_high_Enhance_WT
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')  # 10000
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('--gpu_device', type=str, default='0',
                        help='choose which GPU device you want to use')
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


# 计算平均值
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


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def joint_loss(pred, mask):  # criterion==>joint_loss
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')#reduction='mean'
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  # reduction='mean'
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# 训练函数
def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()
        # 自适应调整学习率
        adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        # compute output 将数据送入网络中计算输出
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += joint_loss(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = joint_loss(output, target)
            iou = iou_score(output, target)


        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()  # 清除上一次所求出的梯度
        loss.backward()  # 误差反向传播
        optimizer.step()  # 优化器开始工作

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    # 保存模型
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoint/%s/' % args.Model + '%d.pth' % (epoch + 1))
        print('[Saving Snapshot]', 'checkpoint/%s/' % args.Model + '%d.pth' % (epoch + 1))

    return log


# 验证
def validate(args, val_loader, model, criterion, epoch: int):
    losses = AverageMeter()
    ious = AverageMeter()
    # Dice accumulators for 3 classes
    dice1_sum = 0.0
    dice2_sum = 0.0
    dice3_sum = 0.0
    dice_count = 0

    # switch to evaluate mode
    model.eval()
    # toggle stage probe only on viz epochs to avoid extra compute
    if hasattr(model, 'set_stage_probe'):
        model.set_stage_probe(enabled=(epoch % 10 == 0), keep=8)
    # clear previous ROI/stage caches so we only aggregate this epoch
    if hasattr(model, 'pop_roi_debug'):
        _ = model.pop_roi_debug()
    if hasattr(model, 'pop_stage_probe'):
        _ = model.pop_stage_probe()

    viz_input0 = None
    viz_target0 = None

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # stash first batch for epoch-wise ROI visualization every 10 epochs (from 0)
            if viz_input0 is None and (epoch % 10 == 0):
                viz_input0 = input.detach().clone()
                viz_target0 = target.detach().clone()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += joint_loss(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = joint_loss(output, target)
                iou = iou_score(output, target)
                output = torch.sigmoid(output).data.cpu().numpy()
                output[output > 0.5] = 1
                output[output <= 0.5] = 0
                tgt_np = target.detach().cpu().numpy()
                for j in range(output.shape[0]):
                    dice_1 = dice_coef(output[j, 0, :, :], tgt_np[j, 0, :, :])
                    dice_2 = dice_coef(output[j, 1, :, :], tgt_np[j, 1, :, :])
                    dice_3 = dice_coef(output[j, 2, :, :], tgt_np[j, 2, :, :])
                    dice1_sum += float(dice_1)
                    dice2_sum += float(dice_2)
                    dice3_sum += float(dice_3)
                dice_count += output.shape[0]

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    dice1_avg = (dice1_sum / dice_count) if dice_count > 0 else 0.0
    dice2_avg = (dice2_sum / dice_count) if dice_count > 0 else 0.0
    dice3_avg = (dice3_sum / dice_count) if dice_count > 0 else 0.0
    dice_mean = (dice1_avg + dice2_avg + dice3_avg) / 3.0
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice1', dice1_avg),
        ('dice2', dice2_avg),
        ('dice3', dice3_avg),
        ('dice_mean', dice_mean),
    ])

    return log


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def main():
    # 先定义一些常用变量
    args = parse_args()
    # args.dataset = "datasets"
    # root_path = '/mnt/sdc/ljc_cnu/MICCAI19_all'
    root_path = '/mnt/sdn/data/BraTS2023_new/'
    homology_use = 0

    # 打印出所有的参数
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')


    with open(f'checkpoint/{args.Model}/args.txt', 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, f'checkpoint/{args.Model}/args.pkl')

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # Data loading code
    train_img_paths = glob(rf'{root_path}/train/Image/*')  # 原始切片路径
    train_mask_paths = glob(rf'{root_path}/train/Mask/*')
    train_mask_paths.sort()
    train_img_paths.sort()
    # print(len(img_paths))

    # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #     train_test_split(img_paths, mask_paths, test_size=0.1, random_state=41)

    val_img_paths = glob(rf'{root_path}/val/Image/*')  # 原始切片路径
    val_mask_paths = glob(rf'{root_path}/val/Mask/*')
    val_mask_paths.sort()
    val_img_paths.sort()
    print("train_num:%s" % str(len(train_img_paths)))
    #    print(train_img_paths)
    print("val_num:%s" % str(len(val_img_paths)))
    # print(val_img_paths)
    # text_save('val20_img_paths.txt', val_img_paths)
    # text_save('val20_mask_paths.txt', val_mask_paths)

    # create model
    print("=> creating model %s" % args.arch)
    # model = VSSM().cuda()
    model = UNetRetNet(
        in_ch=args.input_channels,
        num_classes=3,
        base_ch=32,
        retention_heads=4,
        retention_dim=64,
        retention_at='bottleneck',  # 并行插入点（保留原有 RetNet）
        dropout=0.0,
        retention_window=8,
        retention_shift=0,  # 若想 Swin shift: 设为 8//2=4
        ret_replace_at=[ 'enc4', 'bottleneck']  # 新增：把这些 block 的第2个卷积替换为 RetNet
    ).to(device)

    # Enable ROI debug cache for metrics/visualization
    if hasattr(model, 'set_roi_debug'):
        model.set_roi_debug(True, keep=64)

    model.load_state_dict(torch.load(f'checkpoint/{args.Model}/modelbest.pth'))
    model.to(device)

    # model.load_state_dict(torch.load('checkpoint/unet/Br20high_Enhance_Unet_woDS/modelbest.pth'))#checkpoint/%s/model_best.pth
    # model.eval()
    print(count_params(model))

    # 选择设置优化器
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # 给训练集、验证集赋值
    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    # ################### only flair + t2 + seg WT
    # train_dataset = SomeDataset(args, train_img_paths, train_mask_paths, args.aug)
    # val_dataset = SomeDataset(args, val_img_paths, val_mask_paths)
    # 数据读取接口
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou',
        'dice1', 'dice2', 'dice3', 'dice_mean',
        'roi_valid', 'roi_mask_mean', 'roi_mask_max'
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion, epoch)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - dice1 %.4f - dice2 %.4f - dice3 %.4f - dice_mean %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice1'], val_log['dice2'], val_log['dice3'], val_log['dice_mean']))
        # 如果取消验证部分使用下面保存训练中的loss iou变换
        # print('loss %.4f - iou %.4f '
        #       % (train_log['loss'], train_log['iou']))
        # tmp = pd.Series([
        #     epoch,
        #     args.lr,
        #     train_log['loss'],
        #     train_log['iou'],
        # ], index=['epoch', 'lr', 'loss', 'iou'])
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice1'],
            val_log['dice2'],
            val_log['dice3'],
            val_log['dice_mean'],
            val_log.get('roi_valid', np.nan),
            val_log.get('roi_mask_mean', np.nan),
            val_log.get('roi_mask_max', np.nan),
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou', 'dice1', 'dice2', 'dice3', 'dice_mean',
                  'roi_valid', 'roi_mask_mean', 'roi_mask_max'])
        log = pd.concat([log, pd.DataFrame([tmp])], ignore_index=True)
        log.to_csv('checkpoint/%s/log.csv' % args.Model, index=False)

        trigger += 1

        if val_log['dice_mean'] > best_dice:
            torch.save(model.state_dict(), 'checkpoint/%s/modelbest.pth' % args.Model)
            best_dice = val_log['dice_mean']
            print("=> saved best model")
            trigger = 0
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
