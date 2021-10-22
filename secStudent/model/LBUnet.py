import torch
from torch import nn

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class ShuffleUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, bridge=True):
        super(ShuffleUpBlock, self).__init__()
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
        )
        self.up_p = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                        )
        if bridge:
            self.conv_block = HiConvBlock(out_size * 3, out_size, False, relu_slope)
        else:
            self.conv_block = HiConvBlock(out_size * 2, out_size, False, relu_slope)


    def forward(self, x, bridge=None):
        up = torch.cat([self.up(x), self.up_p(x)], dim=1)
        if bridge != None:
            out = torch.cat([up, bridge], 1)
            out = self.conv_block(out)
        else:
            out = self.conv_block(up)
        return out
class HiConvBlock(nn.Module):
    def __init__(self, inc, ouc, downsample, relu_slope, use_HIN=False):
        super(HiConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(inc, ouc, 1, 1, 0)

        self.conv_1 = nn.Conv2d(inc, ouc, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(ouc, ouc, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(ouc // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(ouc, ouc, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class LBUnet(nn.Module):
    def __init__(self, inc=3, ouc=3, width=64, transblock=[4, 2, 2, 1, 1]):
        super(LBUnet, self).__init__()
        self.conv1 = nn.Conv2d(inc, width, kernel_size=7, stride=2, padding=3)
        self.conv2 = HiConvBlock(width, width * 2, True, 0.2, True)

        self.down1 = HiConvBlock(width * 2, width * 4, downsample=True, relu_slope=0.2, use_HIN=True)
        self.down2 = HiConvBlock(width * 4, width * 4, downsample=True, relu_slope=0.2, use_HIN=True)

        self.trans = nn.ModuleList()
        pre = width * 4
        for i in [4, 4, 2, 1, 1]:
            self.trans.append(HiConvBlock(pre, width * i, downsample=False, relu_slope=0.2, use_HIN=True))
            pre = width * i

        self.up1 = ShuffleUpBlock(width * 16, width * 4, 0.2)
        self.up2 = ShuffleUpBlock(width * 4, width * 1, 0.2)

        self.skipconv1 = nn.Conv2d(width * 4, width * 4, 3, 1, 1)
        self.skipconv2 = nn.Conv2d(width * 4, width * 1, 3, 1, 1)
        self.skipconv3 = nn.Conv2d(width * 2, width * 1, 3, 1, 1)


        self.block = HiConvBlock(width, width *4, False, 0.2, True)
        self.up3 = ShuffleUpBlock(width * 4, width * 1, 0.2, bridge=True)
        self.up4 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(width // 4, ouc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x, down = self.conv2(x)

        x, down1 = self.down1(x)
        x, down2 = self.down2(x)

        map_list = [x]
        for block in self.trans:
            x = block(x)
            map_list.append(x)

        x = torch.cat(map_list, dim=1)
        x = self.up1(x,self.skipconv1(down2))
        x = self.up2(x,self.skipconv2(down1))

        x = self.block(x)
        x = self.up3(x, self.skipconv3(down))
        x = self.up4(x)

        return x

if __name__ == "__main__":
    model = LBUnet(4, 3)
    print(model)
    print(model(torch.rand([1, 4, 128, 128])).shape)











