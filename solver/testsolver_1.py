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

