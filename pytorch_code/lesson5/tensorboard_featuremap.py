#-*-coding:utf-8-*-
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torchvision.utils as vutils
writer = SummaryWriter(comment='test_your_comment', filename_suffix='_test_your_filename_suffix')

path_img = './lena.png'
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]

norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform
])
img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
img_tensor.unsqueeze_(0)

alexnet = models.alexnet(pretrained=True)

convlayer1 = alexnet.features[0]
fmap_1 = convlayer1(img_tensor)

# print(alexnet)
fmap_1.transpose_(0, 1)
fmao_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

writer.add_image('feature map in conv1', fmao_1_grid, global_step=322)
writer.close()