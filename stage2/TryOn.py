# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
import argparse
from torchvision import models
import os
from collections import OrderedDict

from networks import save_checkpoint, VGGLoss

from utils import *

from Unet_Dilation import UNet
class cyclegan(nn.Module):
    def __init__(self, opt):
        '''
        opt, in_channels=19, depth=4,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat')
        :param opt:
        '''
        super(cyclegan, self).__init__()
        self.Generator = UNet(22,4)
        self.Discriminator = NLayerDiscriminator()
        self.PairDis = PairDiscriminator()

        self.criterionGAN = GANLoss("lsgan")
        self.PairGAN = GANLoss("lsgan")
        self.loss_1 = nn.L1Loss()
        self.loss_2 = nn.MSELoss()
        self.optimizer_D = torch.optim.Adam(self.Discriminator.parameters(),
                                            lr=opt.lr, betas=(0.5, 0.999))

        self._optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=opt.lr,
                                             betas=(0.5, 0.999))
        self.vgg_loss = VGGLoss()
        self.content_loss = Content_loss()

    def set_input(self, inputs):
        self.cloth = inputs["cloth"].cuda()
        self.real_A = inputs["c_image"].cuda()
        self.real_B = inputs["t_image"].cuda()
        self.c_bodyshape = inputs["c_shape_sample"].cuda()
        self.t_bodyshape = inputs["Pre_t_shape"].cuda()
        self.c_pose = inputs["c_pose"].cuda()
        self.t_pose = inputs["t_pose"].cuda()
        self.t_cloth_mask = inputs["t_upper_mask"].cuda()
        self.c_cloth = inputs["c_upper_cloth"].cuda()
        self.warp_cloth = inputs["warp_cloth"].cuda()



    def forward(self):
        '''
        return rough, mask, result
        :return:
        def forward(self, x, cloth, head, c_mask):
        '''
        body_A = torch.cat((self.c_bodyshape, self.c_pose), 1)
        body_B = torch.cat((self.t_bodyshape, self.t_pose), 1)
        self.rough_B, self.attention_B, self.gen_B = self.Generator(body_B, self.warp_cloth, self.real_A)
        self.rough_A, self.attention_A, self.gen_A = self.Generator(body_A, self.c_cloth, self.gen_B)


    def backward_D_basic(self, netD, real, fake):

        # Real
        pred_real = netD(real)
        loss_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_fake = self.criterionGAN(pred_fake, False)
        # Fake
        loss_D = loss_fake + loss_real

        return loss_D


    def backward_D(self):

        self.loss_D_B = self.backward_D_basic(self.Discriminator, self.real_B, self.gen_B)
        self.loss_D_A = self.backward_D_basic(self.Discriminator, self.real_A, self.gen_A)
        self.loss_D = self.loss_D_B + self.loss_D_A
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_B = self.criterionGAN(self.Discriminator(self.gen_B), True)
        self.loss_G_A = self.criterionGAN(self.Discriminator(self.gen_A), True)

        self.t_cloth_mask = self.t_cloth_mask.view(-1, 1, 256, 192)
        self.mask_loss = self.loss_1(self.attention_B, self.t_cloth_mask)
        self.smooth_loss = self.loss_smooth(self.attention_B) * 0.0001
        self.attention_loss = self.mask_loss + self.smooth_loss
        self.c_loss = self.content_loss(self.real_B, self.gen_B)
        self.cyc_loss = self.content_loss(self.real_A, self.gen_A)

        self.loss_G = self.attention_loss + self.c_loss * 5 + self.cyc_loss * 5 + \
                      self.loss_G_B + self.loss_G_A

        self.loss_G.backward()


    def loss_smooth(self, mat):

        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def current_results(self):
        loss_dict = OrderedDict(
            [('G_loss', self.loss_G),
             ('content_loss', self.c_loss),

             ('gen_B', self.gen_B),
             ('con_img', self.real_A),
             ('cloth', self.cloth),
             ('tar_img',self.real_B),

             ]
        )
        return loss_dict


    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def optimize_parameters(self):

        self._B = self.real_B.size(0)
        self.forward()
        #
        # self.set_requires_grad([self.Discriminator], True)
        #
        # # self.D_loss = self.backward_D()
        # self.optimizer_D.zero_grad()
        # self.backward_D()
        # self.optimizer_D.step()
        #
        # self.set_requires_grad([self.Discriminator], False)

        self._optimizer_G.zero_grad()
        self.backward_G()
        self._optimizer_G.step()


class Content_loss(nn.Module):
    def __init__(self):
        super(Content_loss,self).__init__()
        self.l1loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()

    def forward(self, inputA, inputA_hat):
        total = self.l1loss(inputA_hat, inputA) + self.vgg_loss(inputA_hat, inputA)
        return total


