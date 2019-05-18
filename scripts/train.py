from __future__ import print_function, division

import argparse
import os
import sys
import time
import gc
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from models import __models__, model_loss
from utils import *
from utils.data import __datasets__

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--log_dir', default='runs', help='the directory to save logs')
parser.add_argument('--ckpt_dir', help='the directory to save and load the weights in/from checkpoints')
parser.add_argument('--pretrained', help='specify the file that has been pretrained')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

opt = parser.parse_args()
print(opt)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(opt.log_dir)

# Initialize model
model = __models__[opt.model](opt.maxdisp)
print('# model parameters:', sum(param.numel() for param in model.parameters()),flush=True)

# find gpu devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_nums = torch.cuda.device_count()

# Use all GPUs by default
if device_nums > 1:
    model = torch.nn.DataParallel(model, device_ids=range(device_nums))
    # generator = torch.nn.DataParallel(generator, device_ids=range(device_nums))
    # discriminator = torch.nn.DataParallel(discriminator, device_ids=range(device_nums))

# Set models to gpu
model = model.to(device)
# generator = generator.to(device)
# discriminator = discriminator.to(device)

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerG = optim.Adam(generator.parameters())
# optimizerD = optim.Adam(discriminator.parameters())

# prepare data
dataset = __datasets__[opt.dataset]
train_dataset = dataset(opt.datapath, opt.trainlist, True)
test_dataset = dataset(opt.datapath, opt.testlist, False)
train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, opt.test_batch_size, shuffle=False, num_workers=0, drop_last=False)

# load checkpoints
start_epoch = 0
if opt.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(opt.ckpt_dir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(opt.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif opt.pretrained:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(opt.pretrained))
    state_dict = torch.load(opt.pretrained)
    weights = OrderedDict()
    for k, v in state_dict['model'].items():
        weights[k.split('module.')[-1]] = v
    model.load_state_dict(weights)
print("start at epoch {}".format(start_epoch))

# loss function
# generator_criterion = GeneratorLoss()
# generator_criterion.to(device)



#  Training
def train():
    for epoch_idx in range(start_epoch, opt.epochs):
        # schedule learning rate
        adjust_learning_rate(optimizer, epoch_idx, opt.lr, opt.lrepochs)

        # training
        for batch_idx, sample in enumerate(train_loader):
            global_step = len(train_loader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % opt.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, opt.epochs,
                                                                                       batch_idx,
                                                                                       len(train_loader), loss,
                                                                                       time.time() - start_time),flush=True)
        # saving checkpoints
        if (epoch_idx + 1) % opt.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(opt.ckpt_dir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(test_loader):
            global_step = len(test_loader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % opt.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, opt.epochs,
                                                                                     batch_idx,
                                                                                     len(test_loader), loss,
                                                                                     time.time() - start_time),flush=True)
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(train_loader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < opt.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < opt.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()