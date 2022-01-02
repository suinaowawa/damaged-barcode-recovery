import torch
import torch.nn as nn
from loss import *
import os
from utils import *
from tqdm.autonotebook import tqdm
import torchvision as tv


class Trainer():
    def __init__(self, opt, model, optimizer, lr_schedule, train_data_loader,
                 valid_data_loader=None, start_epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.opt = opt

        self.cur_epoch = start_epoch

        self.netg = model
        self.optimizer_g = optimizer
        self.scheduler_g = lr_schedule

        self.criterion = nn.BCELoss()
        self.contrast_criterion = nn.MSELoss()

    def train(self):
        if not os.path.exists(self.opt.work_dir):
            os.makedirs(self.opt.work_dir)

        true_labels = torch.ones(self.opt.batch_size)
        fake_labels = torch.zeros(self.opt.batch_size)

        if self.opt.use_gpu:
            self.criterion.cuda()
            self.contrast_criterion.cuda()
            true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()

        for epoch in range(self.opt.max_epoch):
            progressbar = tqdm(self.train_data_loader)
            d_loss = AverageMeter()
            g_loss = AverageMeter()
            c_loss = AverageMeter()
            s_loss = AverageMeter()
            for ii, (imgs, _) in enumerate(progressbar):
                normal, defect, target = imgs
                if self.opt.use_gpu:
                    normal = normal.cuda()
                    defect = defect.cuda()
                    target = target.cuda()

                # train generator
                self.optimizer_g.zero_grad()
                output = self.netg(defect)
                fake_img = output
            
                error_c = self.contrast_criterion(fake_img, normal)
                losses = self.opt.contrast_loss_weight * error_c
                losses.backward()
                self.optimizer_g.step()
                # g_loss.update(error_g)
                c_loss.update(self.opt.contrast_loss_weight * error_c)

                if self.opt.debug:
                    if not os.path.exists(self.opt.save_path):
                        os.makedirs(self.opt.save_path)

                    imgs = torch.cat((defect, fake_img), 0)
                    tv.utils.save_image(imgs, os.path.join(self.opt.save_path, 'train', '{}_defect_repair.jpg'.format(ii)),
                                        normalize=True,
                                        range=(-1, 1))

                progressbar.set_description(
                    'Epoch: {}. Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}. Contrast loss: {:.5f}. Segmentation loss: {:.5f}'.format(
                        epoch, ii, d_loss.getavg(), g_loss.getavg(), c_loss.getavg(), s_loss.getavg()))

            self.scheduler_g.step(epoch=epoch)

            if self.opt.validate:
                self.validate()

            if (epoch + 1) % self.opt.checkpoint_interval == 0:
                state_g = {'net': self.netg.state_dict(), 'optimizer': self.optimizer_g.state_dict(), 'epoch': epoch}
                print('saving checkpoints...')
                torch.save(state_g, os.path.join(self.opt.work_dir, f'g_barcode_e{epoch + 1}.pth'))

    def validate(self):
        self.netg.eval()

        progressbar = tqdm(self.valid_data_loader)
        for ii, (imgs, _) in enumerate(progressbar):
            normal, defect, target = imgs
            if self.opt.use_gpu:
                normal = normal.cuda()
                defect = defect.cuda()
                target = target.cuda()
            repair = self.netg(defect)
            repaired_img = repair

            if self.opt.debug:
                imgs = torch.cat((defect, repaired_img), 0)
                tv.utils.save_image(imgs, os.path.join(self.opt.save_path, 'val', '{}_defect_repair.jpg'.format(ii)),
                                    normalize=True,
                                    range=(-1, 1))
            
