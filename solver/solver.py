#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2020-12-03 22:02:20
@Description: file content
'''
import os, importlib, torch, shutil
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from importlib import import_module
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
import time
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
# progress_bar = Progress(
#     TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
#     BarColumn(),
#     MofNCompleteColumn(),
#     TextColumn("•"),
#     TimeElapsedColumn(),
#     TextColumn("•"),
#     TimeRemainingColumn(),
# )
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class CVLoss(nn.Module):
    def __init__(self, loss_weight=1.0,reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,logits):
        # print(torch.mean(logits,dim=1).shape)
        cv = torch.std(logits,dim=1)/torch.mean(logits,dim=1)
        # print(cv)
        return self.loss_weight*torch.mean(cv**2)
class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        # assert (self.cfg['data']['n_colors']==32)
        self.model = net(
            # num_channels = 5,
             dim =self.cfg['data']['n_colors'],
            # base_filter=64,
            # args = self.cfg
        )
        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])
        # self.loss2 = make_loss(self.cfg['schedule']['loss'])
        self.gate_loss = CVLoss()
        # self.hf_loss = HighFrequencyLoss()
        # self.weight_loss = torch.nn.Parameter(torch.tensor(0.01))
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,500,0.5,last_epoch=-1) #torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,20,0)
        #self.vggloss = make_loss('VGG54')
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,50,1e-5)
        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upsacle']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter(self.cfg['log_dir']+ str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_yml(cfg, os.path.join(self.cfg['log_dir'] + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self):
        # summaries(model, grad=True)
        with Progress(
                *Progress.get_default_columns(),
                # TextColumn(),
                # BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                # TextColumn("Batch loss {:.4f}".format(loss.item()))
        ) as progress:
            task1 = progress.add_task("[yellow]Initial Training Epoch: [{}/{}]".format(self.epoch, self.nEpochs), total=len(self.train_loader))
            # while not progress.finished:
                # progress.update(task1, advance=0.5)

                # time.sleep(0.02)
        # with tqdm(total=len(self.train_loader), miniters=1,
        #         desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:


            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                #ms_image, lms_image, pan_image, bms_image, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
                # gt, pan, lms, ms = batch[0], batch[3], batch[1], batch[4]
                gt, pan, lms = batch[0], batch[3], batch[4]
                if self.cuda:
                    # ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])
                    # gt, pan, lms, ms = gt.cuda(self.gpu_ids[0]), pan.cuda(self.gpu_ids[0]), lms.cuda(self.gpu_ids[0]), ms.cuda(self.gpu_ids[0])
                    gt, pan, lms= gt.cuda(self.gpu_ids[0]), pan.cuda(self.gpu_ids[0]), lms.cuda(self.gpu_ids[0])
                self.optimizer.zero_grad()               
                self.model.train()

                # y = self.model(lms_image, bms_image, pan_image)
                # sr = self.model(lms,pan)
                sr,spa_cof,spe_cof = self.model(lms, pan)
                #
                # sr_fft = torch.fft.fft2(sr)
                # gt_fft = torch.fft.fft2(gt)

                spa_gateloss = self.gate_loss(spa_cof)
                spe_gateloss = self.gate_loss(spe_cof)
                # hf_loss = self.hf_loss(sr,gt)
                loss = self.loss(sr, gt) + (spa_gateloss + spe_gateloss)*0.3
                # loss = self.loss(sr, gt)
                epoch_loss += loss.data
                #epoch_loss = epoch_loss + loss.data + vgg_loss.data

                progress.update(task1, advance=1)
                time.sleep(0.02)

                # t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                # t.update()

                loss.backward()
                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                # total_grad_norm = 0
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         param_norm = param.grad.data.norm(2)
                #         total_grad_norm += param_norm.item() ** 2
                #         print(f'Layer: {name}, Gradient Norm: {param_norm.item():.4f}')
                # total_grad_norm = total_grad_norm ** 0.5
                # print(f'Total Gradient Norm: {total_grad_norm:.4f}')
                # if self.cfg['schedule']['gclip'] > 0:
                #     nn.utils.clip_grad_norm_(
                #         self.model.parameters(),
                #         self.cfg['schedule']['gclip']
                #     )
                self.optimizer.step()
            self.scheduler.step()
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            # self.writer.add_image('image1', ms_image[0], self.epoch)
            # self.writer.add_image('image2', y[0], self.epoch)
            # self.writer.add_image('image3', pan_image[0], self.epoch)
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.6f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with Progress(
                *Progress.get_default_columns(),
                # BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                # TextColumn("Batch loss {:.4f}".format(loss.item()))
        ) as progress:
            task1 = progress.add_task("[blue]Initial Val Epoch: [{}/{}]".format(self.epoch, self.nEpochs), total=len(self.val_loader))
        # with tqdm(total=len(self.val_loader), miniters=1,
        #         desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list = [], []
            val_epoch_loss = 0
            for iteration, batch in enumerate(self.val_loader, 1):
                
                gt, pan, lms = batch[0], batch[3], batch[4]
                if self.cuda:
                    gt, pan, lms,  = gt.cuda(self.gpu_ids[0]), pan.cuda(self.gpu_ids[0]), lms.cuda(self.gpu_ids[0])

                self.model.eval()
                with torch.no_grad():
                    sr,_,_ = self.model(lms, pan)
                    #loss = self.loss(y, ms_image)
                    # loss = criterion(sr, gt)
                    # srfft = torch.fft.rfft2(sr)
                    # gtfft = torch.fft.rfft2(gt)

                    loss = self.loss(sr, gt)
                    val_epoch_loss += loss.data

                progress.update(task1, advance=1)
                time.sleep(0.02)
                # t1.set_postfix_str('n:Batch loss: {:.4f}'.format(loss.item()))
                # t1.update()
            # self.records['Epoch'].append(self.epoch)
            # self.records['PSNR'].append(np.array(psnr_list).mean())
            # self.records['SSIM'].append(np.array(ssim_list).mean())
            self.records['Val_Loss'].append(val_epoch_loss / len(self.val_loader))
            save_config(self.log_name,
                        'Val Epoch {}: Loss={:.6f}'.format(self.epoch, self.records['Val_Loss'][-1]))
            self.writer.add_scalar('Val_Loss_epoch', self.records['Val_Loss'][-1], self.epoch)
            # self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            # cudnn.benchmark = True
            cudnn.benchmark = False
            cudnn.deterministic = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)

            torch.cuda.set_device(self.gpu_ids[0]) 
            self.loss = self.loss.cuda(self.gpu_ids[0])
            self.gate_loss = self.gate_loss.cuda(self.gpu_ids[0])
            # self.hf_loss = self.hf_loss.cuda(self.gpu_ids[0])
            # self.loss2 = self.loss.cuda(self.gpu_ids[0])
            #self.vggloss = self.vggloss.cuda(self.gpu_ids[0])
            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), '{}.pth'.format(self.epoch)))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                self.eval()
                if self.epoch % 10 == 0:
                    self.save_checkpoint()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.log_name, 'Training done.')
