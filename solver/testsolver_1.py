#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2021-01-19 21:00:18
@Description: file content
'''
from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
from data.load_test_data import load_h5py_with_hp, load_h5py
import scipy.io as sio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)

        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net(
            # num_channels=5,
            dim=self.cfg['data']['n_colors'],
            # base_filter=64,
            # args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            # cudnn.benchmark = False

            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >= 0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0])

            self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        # load data
        self.model.eval()
        # ms, _, pan, _ = load_h5py_with_hp(self.cfg['test']['data_dir'])
        ms, _, pan = load_h5py(self.cfg['test']['data_dir'])
        # print("start panshape", pan.shape) # torch.Size([20, 1, 512, 512]) full 512 ruduce 256
        # print("start msshape", ms.shape) # torch.Size([20, 8, 128, 128]) full 128 reduce 64
        for k in range(20):
            with torch.no_grad():
                MS = ms[k,:,:,:].cuda().unsqueeze(dim=0).float()
                # LMS = lms[k,:,:,:].cuda().unsqueeze(dim=0).float()
                PAN = pan[k,0,:,:].cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()
                output,_,_  = self.model(MS,PAN)
                output = torch.clamp(output, 0, 1)
                output = torch.squeeze(output*2047).permute(1,2,0).cpu().detach().numpy()

                save_name = os.path.join(self.cfg['test']['save_dir'],
                                                  "output_" + self.cfg['test']['data_type'] + "Exm_" + str(k) + ".mat")
                sio.savemat(save_name, {'sr': output})

        # get size
        # image_num, C, h, w = ms.shape  # h w 128 128
        # _, _, H, W = pan.shape  # H W  512 512
        # cut_size = 64  # must be divided by 4, we recommand 64
        # ms_size = cut_size // 4  # 16
        # pad = 4  # must be divided by 4
        # edge_H = cut_size - (H - (H // cut_size) * cut_size)  # 64 - (512 - (512//64) * 64) = 64
        # edge_W = cut_size - (W - (W // cut_size) * cut_size)  # 64 - (512 - (512//64) * 64) = 64
        #
        # for k in range(image_num):
        #     with torch.no_grad():
        #         x1, x2 = ms[k, :, :, :], pan[k, 0, :, :]
        #         # x1 torch.Size([8, 128, 128]) x2 torch.Size([512, 512])
        #         x1 = x1.cuda().unsqueeze(dim=0).float()
        #         x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()
        #         # x1 torch.Size([1, 8, 128, 128]) x2 torch.Size([1, 1, 512, 512])
        #
        #         x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4).cuda(
        #             self.gpu_ids[0])  # 128 + 4//2 + 64//4 = 146  (1 , 8 , 146, 146)
        #         x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W).cuda(
        #             self.gpu_ids[0])  # 512 + 4*2 + 64 = 584 (1 , 1, 584 ,584)
        #         x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')  # (1,8,130,130)
        #         x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')  # (1,1,520,520)
        #         x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
        #         x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
        #         output = torch.zeros(1, C, H + edge_H, W + edge_W).cuda(self.gpu_ids[0])  # (1,8,512+64,512+64)
        #
        #         scale_H = (H + edge_H) // cut_size  # 576 // 64 = 9
        #         scale_W = (W + edge_W) // cut_size
        #         for i in range(scale_H):
        #             for j in range(scale_W):
        #                 MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
        #                      j * ms_size: (j + 1) * ms_size + pad // 2]
        #                 PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
        #                       j * cut_size: (j + 1) * cut_size + 2 * pad]
        #                 # print('MS PAN', MS.shape, PAN.shape) # 1 8 18 18   1 1 72 72
        #                 sr = self.model(MS, PAN)
        #                 # print('sr', sr.shape) # 1 8 72 72
        #                 sr = torch.clamp(sr, 0, 1)
        #                 output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
        #                     sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * 2047.
        #                 # print('output_', output.shape)
        #         # print('output1',output.shape)# 1 8 320 320
        #         output = output[:, :, :H, :W]
        #         # print('output2', output.shape) # 1 8 256 256
        #         output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
        #         save_name = os.path.join(self.cfg['test']['save_dir'],
        #                                  "output_" + self.cfg['test']['data_type'] + "Exm_" + str(k) + ".mat")
        #         sio.savemat(save_name, {'sr': output})

    # def eval(self):
    #     self.model.eval()
    #     avg_time= []
    #     for batch in self.data_loader:
    #         ms_image, lms_image, pan_image, bms_image, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), (batch[4])
    #         if self.cuda:
    #             lms_image = lms_image.cuda(self.gpu_ids[0])
    #             pan_image = pan_image.cuda(self.gpu_ids[0])
    #             bms_image = bms_image.cuda(self.gpu_ids[0])
    #
    #         t0 = time.time()
    #         with torch.no_grad():
    #             prediction = self.model(lms_image, bms_image, pan_image)
    #
    #         t1 = time.time()
    #
    #         if self.cfg['data']['normalize']:
    #             lms_image = (lms_image+1) /2
    #             pan_image = (pan_image+1) /2
    #             bms_image = (bms_image+1) /2
    #
    #         print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
    #         avg_time.append(t1 - t0)
    #         self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
    #         self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
    #     print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        # print((save_img.max()))
        # save img
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_fn = save_dir + '/' + img_name
        save_img = np.uint8(save_img * 255).astype('uint8')  #
        # print(save_img.max())
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)

    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':
            # self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
            # self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
            #     num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':
            self.dataset = get_eval_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                                          num_workers=self.cfg['threads'])
            self.eval()
        else:
            raise ValueError('Mode error!')
