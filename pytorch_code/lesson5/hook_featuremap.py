#-*-coding:utf-8-*-
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from PIL import  Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
writer = SummaryWriter(comment='test_your_comment', filename_suffix='_test_your_filename_suffix')
path_img = './lena.png'
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]

norm_transforms = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transforms
])

img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)
alexnet = models.alexnet(pretrained=True)
fmap_dict = dict()
for name, sub_module in alexnet.named_modules():
    if isinstance(sub_module, nn.Conv2d):
        key_name = str(sub_module.weight.shape)
        fmap_dict.setdefault(key_name, list())
        n1, n2 = name.split('.')

        def hook_func(m, i, o):
            key_name = str(m.weight.shape)
            fmap_dict[key_name].append(o)

        alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)


output = alexnet(img_tensor)
for layer_name, fmap_list in fmap_dict.items():
    fmap = fmap_list[0]
    # print('1', fmap.shape)
    fmap.transpose_(0, 1)
    # print('2', fmap.shape)
    nrow = int(np.sqrt(fmap.shape[0]))
    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=322)
