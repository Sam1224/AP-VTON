import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from networks import ResidualBlock

from cp_dataset import CPDataset, CPDataLoader
from Affine_TPS import TPS_Affine
import argparse
import os
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=False, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,  out_channels, kernel_size=4, stride=2, padding=1)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2), conv3x3(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.res = ResidualBlock(in_channels,in_channels)
        down = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU(True)]

        self.down = nn.Sequential(*down)

    def forward(self, x):
        res = self.res(x)
        before_pool = res
        x = self.down(before_pool)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        upconv = [upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(True)]

        self.res = ResidualBlock(2 * self.out_channels, 2 * self.out_channels)
        self.conv = conv3x3(2 * self.out_channels, self.out_channels)
        self.upconv = nn.Sequential(*upconv)


    def forward(self, from_down, from_up):
        up = self.upconv(from_up)
        cat = torch.cat((up, from_down), 1)
        res = self.res(cat)
        res = self.conv(res)
        return res



class Bottleneckt(nn.Module):
    '''
        :return upper features, head features,and warped cloth features
    '''
    def __init__(self, opt, TPS=True):
        super(Bottleneckt, self).__init__()
        self.TPS = TPS
        self.cloth_features = Cloth_features(3, 512, 64)
        self.person_paring_features = Person_paring_features(3, 256, 64)
        self.res = ResidualBlock(512, 512)
        self.grid_TPS = Warping(opt)
        self.Affine_TPS = TPS_Affine(opt)

    def forward(self, cloth, head, lower, t_parse, c_mask):
        c_features = self.cloth_features(cloth)
        head_features = self.person_paring_features(head)
        lower_features = self.person_paring_features(lower)
        res_c_fea = self.res(c_features)
        res_cloth = torch.cuda.FloatTensor(res_c_fea)
        c_features = torch.cuda.FloatTensor(c_features)
        if self.TPS:
            warp = self.grid_TPS(res_cloth, t_parse, c_mask)
        else:
            warp = self.Affine_TPS(res_cloth, t_parse, c_mask)

        c_features = c_features + warp
        return torch.cat((head_features, lower_features, c_features),1)

class UNet(nn.Module):

    def __init__(self, opt, in_channels=38, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.Bottle = Bottleneckt(opt)
        self.down_convs = []
        self.up_convs = []
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, start_filts, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(start_filts, affine=True),
            nn.ReLU(inplace=True))
        self.conv_first = conv1

        #cloth-warp
        self.cwarp = Warping(opt, 1, 20)
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        #person-warp
        self.pwarp = Warping(opt, 20, 20)

        self.cloth_person_encoder = []
        for i in range(depth-2):
            mul = 2 ** i
            print(mul)
            encoder = DownConv(start_filts * mul, start_filts * mul * 2)
            self.cloth_person_encoder.append(encoder)

        for i in range(2):
            encoder = DownConv(512, 512)
            self.cloth_person_encoder.append(encoder)

        for i in range(depth-2):
            mul = 2 ** i
            print(mul)
            down_conv = DownConv(start_filts * mul, start_filts * mul * 2)
            self.down_convs.append(down_conv)

        for i in range(2):
            down_conv = DownConv(512, 512)
            self.down_convs.append(down_conv)


        self.bottle = nn.Sequential(ResidualBlock(1024, 1024),
                              conv3x3(1024, 512), nn.BatchNorm2d(512),
                              nn.ReLU(inplace=True))



        for i in range(2):
            up_conv = UpConv(512, 512)
            self.up_convs.append(up_conv)
        outs = 512
        for i in range(depth-2):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # self.up = UpConv(128, 64)
        # self.conv_final = conv1x1(outs, self.num_classes)
        # self.conv_first = nn.ModuleList(self.conv_first)
        self.down_convs = nn.ModuleList(self.down_convs)
        # self.bottle = nn.ModuleList(self.bottle)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.last_layer = nn.Sequential(nn.Conv2d(outs, 3, kernel_size=3, stride=1, padding=1),
                                        nn.Tanh()
        )
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            # init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)




    def forward(self, x, cloth, head, lower, t_parse, c_mask):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        x = self.conv_first(x)
        # print(x.shape)
        y = self.Bottle(cloth, head, lower, c_mask, t_parse)
        # print(y.shape)
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            # print(x.shape)
        
        x = torch.cat((x,y),1)
        x = self.bottle(x)
        print(x.shape)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 1)]
            x = module(before_pool, x)

        x = self.last_layer(x)
        print(x.shape)
        return x

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Try-On")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument("--dataroot", default="/home/a/PycharmProjects/test1")
    parser.add_argument("--datamode", default="blouse&tunics")
    parser.add_argument("--stage", default="Try-On")
    # parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=6)
    parser.add_argument("--fine_height", type=int, default=8)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/a/PycharmProjects/checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=1)
    parser.add_argument("--decay_step", type=int, default=0)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    opt = get_opt()
    print(opt)
    print("named: %s!" % (opt.name))
    model = UNet(opt, 38, 5, merge_mode='concat')
    model.cuda()
    x = Variable(torch.FloatTensor(np.random.random((1, 38, 256, 192)))).cuda()
    y1 = Variable(torch.FloatTensor(np.random.random((1, 3, 256, 192)))).cuda()
    y2 = Variable(torch.FloatTensor(np.random.random((1, 3, 256, 192)))).cuda()
    y3 = Variable(torch.FloatTensor(np.random.random((1, 3, 256, 192)))).cuda()
    y4 = Variable(torch.FloatTensor(np.random.random((1, 20, 256, 192)))).cuda()
    y5 = Variable(torch.FloatTensor(np.random.random((1, 1, 256, 192)))).cuda()
    out = model(x, y1, y2, y3, y4, y5)


