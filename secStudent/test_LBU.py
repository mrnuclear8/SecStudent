import argparse
import itertools
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from model.LBUnet import LBUnet
from model.models import *
from data_processing.datasets import *
from utils import *
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=99, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="cartoonCity_LBU_dataset_256w_key", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
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


cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
# G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_AB = LBUnet(4,3)
G_BA = LBUnet(3,4)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()


if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    # D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    # D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    # G_AB.apply(weights_init_normal)
    # G_BA.apply(weights_init_normal)
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
    transforms.Resize([256,512], Image.BICUBIC),
    # transforms.RandomCrop((512, 256)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def sharp(image):
    image = image.cpu()
    image = image.detach().numpy()
    print(type(image))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
    dst = cv2.filter2D(image, -1, kernel=kernel)
    # dst = torch.from_numpy(dst).cuda()
    print(type(dst))
    return dst

import PIL
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    input_images_path = 'C:/Users/XXX/PycharmProjects/secStudent/images/input_image/people'
    output_iamges_path = 'C:/Users/XXX/PycharmProjects/secStudent/images/output_image'
    key = torch.load('C:/Users/XXX/PycharmProjects/secStudent/saved_key/LBU_key.pth')
    key = key.unsqueeze(0)

    # CUHK_realA_path ='C:/Users/XXX/PycharmProjects/secgan-master/images/CUHK_OP/three_pic_08/realA'
    # CUHK_fakeA_path ='C:/Users/XXX/PycharmProjects/secgan-master/images/CUHK_OP/three_pic_08/fakeB'
    # CUHK_cycleA_path ='C:/Users/XXX/PycharmProjects/secgan-master/images/CUHK_OP/three_pic_08/cycle_A'
    # key = torch.randint(0,255,[1,1,256,512])/255
    images_name_lists = os.listdir(input_images_path)
    


    # print(imgs.shape)
    # exit()
    G_AB.eval()
    G_BA.eval()
    from torch.cuda.amp import autocast
    with autocast():
        for i in tqdm(range(len(images_name_lists))):
            imgs = Image.open(input_images_path+'/'+images_name_lists[i])

            tran = transforms.Compose(transforms_)
            imgs = tran(imgs).unsqueeze(0)


            imgs = torch.cat([imgs, key], dim=1)
            real_A = Variable(imgs.type(Tensor))
            start = time.time()
            fake_B = G_AB(real_A)
            cycle_A = G_BA(fake_B)

            end = time.time()
            print('time:',end-start)
            # #尝试锐化
            # cycle_A = sharp(cycle_A)

            # Arange images along x-axis
            real_A = make_grid(real_A[:,:3,:,:], nrow=1, normalize=True)
            fake_B = make_grid(fake_B, nrow=1, normalize=True)
            cycle_A = make_grid(cycle_A[:,:3,:,:], nrow=1, normalize=True)

            # Arange images along y-axis
            image_grid = torch.cat((real_A, fake_B, cycle_A,), 1)
            save_image(image_grid, output_iamges_path+'/'+images_name_lists[i], normalize=False)
            
            # pic_name = images_name_lists[i].split('.')
            # save_image(real_A, images_root_path + '/' + pic_name[0]+'real_A'+'.png', normalize=False)
            # save_image(fake_B, images_root_path + '/' + pic_name[0]+'fake_B'+'.png', normalize=False)
            # save_image(cycle_A, images_root_path+'/'+ pic_name[0]+'cycle_A'+'.png', normalize=False)

sample_images(1)