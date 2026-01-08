import os, torch, time
from utils.utils import save_config
from data.dataset import data
from data.data import get_data
from torch.utils.data import DataLoader
from data.load_train_data import Dataset_Pro

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        self.timestamp = int(time.time())

        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = Dataset_Pro(cfg['data_dir_train'])
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=self.num_workers,
                                       pin_memory=True, drop_last=True)
        self.val_dataset = Dataset_Pro(cfg['data_dir_eval'])
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=self.num_workers,
                                     pin_memory=True, drop_last=True)



        self.records = {'Epoch': [],  'Loss': [], 'Val_Loss': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            # self.save_records()
            self.epoch += 1
        #self.logger.log('Training done.')
