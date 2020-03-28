#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import numpy as np

log_dir = './train_log/test_log_dir'
writer = SummaryWriter(comment='_scalars', filename_suffix='12345678')
for x in range(100):
    writer.add_scalar('y=pow_2_x', 2 ** x, x)
writer.close()