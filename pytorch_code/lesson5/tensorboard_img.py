#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
np.random.seed(1)

writer = SummaryWriter(comment='test_your_comment', filename_suffix='_test_your_filename_suffix')
fake_img = torch.randn(3, 512, 512)
writer.add_image('fake_img', fake_img, 1)
time.sleep(1)

fake_img = torch.ones(3, 512, 512)
time.sleep(1)
writer.add_image('fake_img', fake_img, 2)

fake_img = torch.ones(3, 512, 512) * 1.1
time.sleep(1)
writer.add_image('fake_img', fake_img, 3)

fake_img = torch.rand(512, 512)
writer.add_image('fake_img', fake_img, 4, dataformats='HW')

fake_img = torch.rand(512, 512, 3)
writer.add_image('fake_img', fake_img, 5, dataformats='HWC')

writer.close()

