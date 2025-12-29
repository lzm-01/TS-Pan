#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2020-11-25 23:00:57
@Description: file content
'''
import os, math, torch,cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F
# from model.deepfuse import MEF_SSIM_Loss
class newLoss(nn.Module):
    def __init__(self,r1=1.1,r2=1000,offset=0.9995):
        super(newLoss, self).__init__()
        self.r1=r1
        self.r2=r2
        self.offset=offset
        print("new loss maked")
    def forward(self, input,target):
        del_x=torch.abs(input-target)
        compare_mut=torch.ones_like(del_x)
        y = ((del_x*255)**self.r1/255)*del_x*1#((torch.minimum(self.offset+del_x,compare_mut))**self.r2)
        return torch.sum(y)



def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], weight_decay=0)
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='sum')
    elif loss_type == "L1":
        loss = nn.L1Loss(size_average=True)
    elif loss_type == "MEF_SSIM":
        loss = MEF_SSIM_Loss()
    elif loss_type == "VGG22":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "VGG54":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "newloss":
        loss = newLoss()

    else:
        raise ValueError
    return loss

def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    # open_type = 'a' if os.path.exists(get_path('/data/lzm/MSDNN/logsmsddn_4_1729262147/records.txt'))else 'w' # get_path('/data/lzm/MSDNN/logs/' + str(time) + '/records.txt')
    open_type = 'a' if os.path.exists(get_path(
        '/data/lzm/Ablation_Sec_wv3/no_ms_hf/logs/' + str(time) + '/records.txt')) else 'w'  # get_path('/data/lzm/MSDNN/logs/' + str(time) + '/records.txt')
    log_file = open(get_path('/data/lzm/Ablation_Sec_wv3/no_ms_hf/logs/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(time, log):
    open_type = 'a' if os.path.exists(get_path('/data/lzm/Ablation_Sec_wv3/no_ms_hf/logs/' + str(time) + '/net.txt')) else 'w'
    log_file = open(get_path('/data/lzm/Ablation_Sec_wv3/no_ms_hf/logs/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')
