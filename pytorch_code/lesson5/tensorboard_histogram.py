#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
writer = SummaryWriter(comment='test_comment', filename_suffix='test_suffix')
for x in range(2):
    np.random.seed(x)
    data_union = np.arange(100)
    data_normal = np.random.normal(size=1000)

    writer.add_histogram('distribution union', data_union, x)
    writer.add_histogram('distribution normal', data_normal, x)
    plt.subplot(121).hist(data_union, label='union')
    plt.subplot(122).hist(data_normal, label='normal')
    plt.legend()
    plt.show()
writer.close()
