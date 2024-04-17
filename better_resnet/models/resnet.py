'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob=0.2, block_size=7, drop_at_inference=False, drop_generator=None):
        super(DropBlock2D, self).__init__()
        self.drop_at_inference = drop_at_inference
        self.drop_prob = drop_prob
        self.block_size = block_size
        if drop_generator is None:
            self.drop_generator = torch.Generator().manual_seed(int(time.time()*10**10))
        else:
            self.drop_generator = drop_generator

    def __str__(self) -> str:
        return f"DropBlock2D(drop_prob={self.drop_prob}, block_size={self.block_size}, seed={self.drop_generator.initial_seed()%100}"
    
    def __repr__(self) -> str:
        return f"DropBlock2D(drop_prob={self.drop_prob}, block_size={self.block_size}, seed={self.drop_generator.initial_seed()%100}"
    

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if (not self.training or self.drop_prob == 0.) and self.drop_at_inference == False:
            return x
        # get gamma value
        gamma = self._compute_gamma(x)

        # if self.drop_at_inference:
        #     filename = "#big.txt"
        # else:
        #     filename = "#small.txt"
        # with open(filename, 'a') as f:
        #     f.write(str(torch.rand([1], generator=self.drop_generator).item())+'\n')
        #     time.sleep(0.1)
        # sample mask
        mask = (torch.rand(x.shape[0], *x.shape[2:], generator=self.drop_generator) < gamma).float()

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)

        # apply block mask
        out = x * block_mask[:, None, :, :]

        # scale output
        out = out * block_mask.numel() / block_mask.sum()

        return out


    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=3, drop_at_inference=False, drop_generator=None):
        super(BasicBlock, self).__init__()
        self.db1 = DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.db2 = DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.db1(self.conv1(x))))
        out = self.bn2(self.db2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=3, drop_at_inference=False, drop_generator=None):
        super(Bottleneck, self).__init__()
        self.db1 = DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.db2 = DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.db3 = DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                DropBlock2D(drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.db1(self.conv1(x))))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn2(self.db2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        out = self.bn3(self.db3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, drop_prob=0.13, block_size=3, drop_at_inference=False, drop_generator=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.drop_prob = drop_prob
        self.drop_at_inference = drop_at_inference
        self.drop_generator = drop_generator
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, drop_prob=0, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, drop_prob=0, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop_prob=drop_prob, block_size=block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, drop_prob=0.12, block_size=3, drop_at_inference=False, drop_generator=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_prob, block_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        initial_out = F.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(initial_out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        avg_pool_out = F.avg_pool2d(layer4_out, 4)
        reshaped_out = avg_pool_out.view(avg_pool_out.size(0), -1)
        final_out = self.linear(reshaped_out)
        return final_out, {'initial_out': initial_out, 'layer1': layer1_out, 'layer2': layer2_out, 'layer3': layer3_out, 'layer4': layer4_out, 'avg_pool_out': avg_pool_out, 'reshaped_out': reshaped_out}


def ResNet18(dropblock_prob=0.11, dropblock_size=3, drop_at_inference=False, drop_generator=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], drop_prob=dropblock_prob, block_size=dropblock_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)


def ResNet34(dropblock_prob=0.11, dropblock_size=3, drop_at_inference=False, drop_generator=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], drop_prob=dropblock_prob, block_size=dropblock_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)



def ResNet50(dropblock_prob=0.11, dropblock_size=3, drop_at_inference=False, drop_generator=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], drop_prob=dropblock_prob, block_size=dropblock_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)


def ResNet101(dropblock_prob=0.11, dropblock_size=3, drop_at_inference=False, drop_generator=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], drop_prob=dropblock_prob, block_size=dropblock_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)


def ResNet152(dropblock_prob=0.11, dropblock_size=3, drop_at_inference=False, drop_generator=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], drop_prob=dropblock_prob, block_size=dropblock_size, drop_at_inference=drop_at_inference, drop_generator=drop_generator)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()