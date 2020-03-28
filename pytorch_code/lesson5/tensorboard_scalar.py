#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
max_epoch = 100
writer = SummaryWriter(comment='test_comment', filename_suffix='test_suffix')
for x in range(max_epoch):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow_2_x', 2 ** x, x)
    writer.add_scalars('data/scalar_group',{'xsinx': x * np.sin(x),
                                            'xcosx': x * np.cos(x)}, x)
writer.close()
