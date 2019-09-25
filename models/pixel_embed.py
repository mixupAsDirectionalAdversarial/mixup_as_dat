'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .resnet import *

class Pixel_Embed_ResNet(nn.Module):
    def __init__(self, embed_size, block, num_blocks, num_classes=10):
        super(Pixel_Embed_ResNet, self).__init__()
        self.in_planes = 64
        print("EMBEDDING SIZE: ", embed_size)
        self.conv1 = conv3x3(3*embed_size,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.pixel_embed = nn.Embedding(256, embed_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5, lamda=None, mix_index=None):

        # convert back to 0-255 pixel values

        # 128 x 3 x 32 x 32
        #x=(x*255).long()

        # 128 x 3 x 32 x 32 x 20
        embedded_x = self.pixel_embed(x)

        # 128 x 60 x 32 x 32 -- where first 20 of each "pixel" is red, next 20 is green, last 20 is blue
        new_shape = torch.tensor(x.size())
        new_shape[1]=-1
        new_shape = tuple(new_shape)
        embedded_concat = embedded_x.permute(0, 1, 4, 2, 3).contiguous().view(new_shape)
        # convert to one-hot
        #inputs_one_hot = torch.eye(256)[x]

        if lamda is None:
            out = embedded_concat
        else:
            out = lamda * embedded_concat + (1 - lamda) * embedded_concat[mix_index]


        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def Pixel_Embed_ResNet18(embed_size):
    return Pixel_Embed_ResNet(embed_size, PreActBlock, [2,2,2,2])


def test_pixel_embed():
    net = Pixel_Embed_ResNet18(20)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_pixel_embed()
