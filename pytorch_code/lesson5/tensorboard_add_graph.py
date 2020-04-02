#-*-coding:utf-8-*-
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch
from torchsummary import summary
writer = SummaryWriter(comment='test_your_comment', filename_suffix='_test_your_filename_suffix')
fake_img = torch.randn(1, 3, 32, 32)
vgg = models.vgg11()
writer.add_graph(vgg, fake_img)
writer.close()

# summary(vgg, (3, 40, 40), device='cpu')
