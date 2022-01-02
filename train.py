import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from model import *
from defect import DefectAdder, NormalizeList, ToTensorList
from utils import *
from loss import *
import os
from trainer import *
import warnings
warnings.filterwarnings("ignore") 

class Config(object):
    data_path = r'images/train'
    val_path = r'images/val'
    save_path = os.getcwd()+'/data/save_path'
    work_dir = os.getcwd()+'\\data\work_dir'

    if not os.path.exists(save_path + '/train'):
        os.makedirs(save_path + '/train')
    if not os.path.exists(save_path + '/val'):
        os.makedirs(save_path + '/val')

    num_workers = 4
    image_size = 128
    batch_size = 16
    max_epoch = 1000
    steps = [40, 80]
    lrg = 1e-4
    lrd = 1e-4
    lrs = 1e-2
    beta1 = 0.5

    nBottleneck = 4000
    nc = 3
    ngf = 64
    ndf = 64
    defect_mode = 'geometry'

    contrast_loss_weight = 1

    # device settings
    use_gpu = True
    gpus = 1
    nodes = 1
    nr = 0

    netd_path = None
    netg_path = None

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    checkpoint_interval = 100

    debug = True
    validate = True


def main(opt):
    
    print('undistributed training')
    train(opt)


def train(opt):
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        DefectAdder(mode=opt.defect_mode, defect_shape=('line',)),
        ToTensorList(),
        NormalizeList(opt.mean, opt.std),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    train_dataloader = DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  drop_last=True)

    if opt.validate:
        val_transforms = tv.transforms.Compose([
            tv.transforms.Resize(opt.image_size),
            tv.transforms.CenterCrop(opt.image_size),
            DefectAdder(mode=opt.defect_mode, defect_shape=('circle', 'square')),
            ToTensorList(),
            NormalizeList(opt.mean, opt.std),
        ])

        val_dataset = tv.datasets.ImageFolder(opt.val_path, transform=val_transforms)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers,
                                    drop_last=True)
    else:
        val_dataloader = None

    map_location = lambda storage, loc: storage
    # netd = Discriminator(opt)
    # netg = Generater(opt)
    netg = U_Net()

    if opt.use_gpu:
        # netd.cuda()
        netg.cuda()

    if opt.netg_path:
        print('loading checkpoint for generator...')
        checkpoint = modify_checkpoint(netg, torch.load(opt.netg_path, map_location=map_location)['net'])
        netg.load_state_dict(checkpoint, strict=False)

    optimizer_g = optim.Adam(netg.parameters(), opt.lrg, betas=(opt.beta1, 0.999))
    # optimizer_d = optim.Adam(netd.parameters(), opt.lrd, betas=(opt.beta1, 0.999))
    # optimizer_s = optim.Adam(nets.parameters(), opt.lrs, betas=(opt.beta1, 0.999))

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
    # scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)
    # scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=opt.steps, gamma=0.1)

    trainer = Trainer(opt, netg, optimizer_g, scheduler_g, train_dataloader, val_dataloader)
    trainer.train()



if __name__ == '__main__':
    opt = Config()
    main(opt)
