import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from torch.backends.cudnn import benchmark
from model.LBUnet import LBUnet
benchmark = True
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model.models import *
from data_processing.datasets import *
from utils import *

from loss_utils.mix_loss import mix_loss, MS_SSIM_L1_LOSS, manhattan_metric
from torch.cuda.amp import autocast

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="cartoonCity_hinet_dataset_256w_key", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10., help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_cycle = MS_SSIM(data_range=1.0, channel=3)
criterion_cycle = mix_loss()
criterion_cycle2 = torch.nn.SmoothL1Loss()
criterion_cycle3 = torch.nn.MSELoss()

criterion_key = manhattan_metric()

# criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
# G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_AB = LBUnet(4, 3)
G_BA = LBUnet(3, 4)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    # criterion_cycle2.cuda()
    # criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    # G_AB.apply(weights_init_normal)
    # G_BA.apply(weights_init_normal)
    # G_AB._initialize()
    # G_BA._initialize()
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize([256, 512], Image.BICUBIC),

    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    dataset=ImageDataset(r"C:\Users\XXX\PycharmProjects\LBU\images\dataset", transforms_=transforms_, unaligned=False),
    batch_size=8,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)
# Test data loader
val_dataloader = DataLoader(
    dataset=ImageDataset(r"C:\Users\XXX\PycharmProjects\LBU\images\dataset", transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    with autocast():
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        cycle_A = G_BA(fake_B)
    # Arange images along x-axis
    real_A = make_grid(real_A[:,:3,:,:], nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A[:,:3,:,:], nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    cycle_A = make_grid(cycle_A[:,:3,:,:], nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, cycle_A, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)




prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)


        G_AB.train()
        G_BA.train()


        # loss_id_A = criterion_identity(G_BA(real_A), real_A)
        # loss_id_B = criterion_identity(G_AB(real_B), real_B)

        # loss_identity = (loss_id_A + loss_id_B) / 2
        loss_identity = 0

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) + 0.1 * criterion_cycle(fake_B, real_B)
        # fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)

        # loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN_BA = criterion_GAN(D_A(recov_A[:,:3,:,:]), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        # loss_GAN = loss_GAN_AB

        # recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A[:,:3,:,:], real_A[:,:3,:,:])
        recov_B = G_AB(recov_A)
        # loss_cycle_B_real = criterion_cycle2(recov_B[1], real_B) + 0.1 * criterion_cycle2(recov_B[0], real_B)
        loss_cycle_B_real = 0
        loss_cycle_B_fake = criterion_cycle2(recov_B, fake_B)
        loss_cycle_B = (loss_cycle_B_real + loss_cycle_B_fake) / 2

        # loss_cycle = 0.75 * loss_cycle_A + 0.25 * loss_cycle_B
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # loss_cycle = loss_cycle_A

        #Key loss
        loss_key = criterion_key(recov_A[:,3:,:,:], real_A[:,3:,:,:])
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity + 0.0001 * loss_key

        loss_G.backward()
        nn.utils.clip_grad_norm_(G_AB.parameters(), max_norm=10.)
        nn.utils.clip_grad_norm_(G_BA.parameters(), max_norm=10.)

        optimizer_G.step()


        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A[:,:3,:,:]), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(recov_A[:,:3,:,:])
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        # loss_D = (loss_D_A + loss_D_B) / 2
        loss_D = loss_D_B


        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f / %f] [G loss: %f, adv: %f / %f, cycle: %f, key: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_A.item(),
                loss_D_B.item(),
                loss_GAN_BA.item(),
                loss_GAN_AB.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                # loss_identity.item(),
                loss_key.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
