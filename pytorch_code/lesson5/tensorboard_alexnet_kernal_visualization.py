#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torchvision.utils as vutils
writer = SummaryWriter(comment='test_your_comment', filename_suffix='_test_your_filename_suffix')
alexnet = models.alexnet(pretrained=True)

kernel_num = -1
vis_max = 1
for sub_module in alexnet.modules():
    if isinstance(sub_module, nn.Conv2d):
        kernel_num += 1
        if kernel_num > vis_max:
            break
        kernels = sub_module.weight
        c_out, c_int, k_w, k_h = tuple(kernels.shape)

        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=322)
        print('{}_convlayer shape:{}'.format(kernel_num, tuple(kernels.shape)))
    writer.close()
