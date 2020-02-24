# coding=utf-8
import torch
import os
import skimage
import torch.nn as nn
from utils import *

ssim = 0
psnr = 0
mse = 0.0

from  skimage import io as io
import numpy as np
vgg_loss = VGGLoss()
loss_1 = nn.L1Loss()
loss1 = 0
vgg = 0
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



        # load data list
pairs = []
c_names = []
t_names = []
test_dataroot = "/media/a/新加卷/work1/Codes/testdata"
result_root = "test/test_Unet_4"
with open("test_pairs.txt", 'r') as f:
    for line in f.readlines():
        pair, c_name, t_name = line.strip().split()
        pairs.append(pair)
        c_names.append(c_name)
        t_names.append(t_name)

for i in range(len(pairs)):
    pair = pairs[i]
    t_name = t_names[i]
    if i % 100 == 0:
        print(i, ssim, loss1, vgg)
    path_real = os.path.join(test_dataroot, pair, t_name + "_target.jpg")
    path_fake = os.path.join(result_root,pair+'.jpg')
    fake = io.imread(path_fake)
    real = io.imread(path_real)
    ssim += skimage.measure.compare_ssim(real, fake, multichannel=True)
    psnr += skimage.measure.compare_psnr(real, fake)
    mse += skimage.measure.compare_mse(real,fake)
    fake = transform(fake).view(1,3,256,192).cuda()
    real = transform(real).view(1,3,256,192).cuda()
    loss1 += loss_1(fake, real)
    vgg += vgg_loss(fake, real)


print("ssim:", ssim / len(pairs))
print("l1:", loss1 / len(pairs))
print("vgg:", vgg / len(pairs))
import torch
# for i in range(images):
#     if i % 100 == 0:
#         print(i, ssim, loss1, vgg)
#     path_fake = os.path.join(path, str(i) + '_gen.jpg')
#     path_real = os.path.join(path, str(i) + '_tar.jpg')
#     fake = io.imread(path_fake)
#     real = io.imread(path_real)
#     ssim += skimage.measure.compare_ssim(real, fake, multichannel=True)
#     psnr += skimage.measure.compare_psnr(real, fake)
#     mse += skimage.measure.compare_mse(real,fake)
#     # fake = torch.from_numpy(fake)
#     # real = torch.from_numpy(real)
#     fake = transform(fake).view(1,3,256,192).cuda()
#     real = transform(real).view(1,3,256,192).cuda()
#     loss1 += loss_1(fake, real)
#     vgg += vgg_loss(fake, real)

# for index, i in enumerate(files):
#     if index % 100 == 0:
#         print(index, ssim, psnr)
#         res = skimage.io.imread(path+i)
#         real = skimage.io.imread('/home/chenzhaozheng/data/test/'+i.split('_')[0]+'/'+i.split('_')[1]+'_'+i.split('_')[2])
#         ssim += skimage.measure.compare_ssim(real, res, multichannel=True)
#         psnr += skimage.measure.compare_psnr(real, res)
#
print("ssim:", ssim * 2 /len(files))
print("l1:", loss1 * 2/len(files))
print("vgg:", vgg * 2/len(files))