# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AllConv implementation (https://arxiv.org/abs/1412.6806)."""
"""https://github.com/google-research/augmix/blob/master/cifar.py
"""
import math
import torch
import torch.nn as nn
from utils import *


def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    else:
        padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


class GELU(nn.Module):

  def forward(self, x):
    return torch.sigmoid(1.702 * x) * x


def make_layers(cfg):
  """Create a single layer."""
  layers = []
  # in_channels = 3
  global in_channels
  for v in cfg:
    if v == 'Md':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
    elif v == 'A':
      layers += [nn.AvgPool2d(kernel_size=8)]
    elif v == 'NIN':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
      layers += [conv2d, nn.BatchNorm2d(in_channels)]
    elif v == 'nopad':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
      layers += [conv2d, nn.BatchNorm2d(in_channels)]
    elif v == 'gelu':
      layers += [GELU()]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      layers += [conv2d, nn.BatchNorm2d(v)]
      in_channels = v
  return nn.Sequential(*layers)


class AllConvNetCBS(nn.Module):
  """AllConvNet main class."""

  def __init__(self, num_classes, args):
    super(AllConvNetCBS, self).__init__()
    
    self.std = args.std
    self.epoch = args.epoch

    global in_channels
    in_channels = 3
    
    self.num_classes = num_classes
    self.width1, w1 = 96, 96
    self.width2, w2 = 192, 192

    # self.features = make_layers(
    #     [w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'])
    
    self.conv1 = make_layers([w1, 'gelu', w1, 'gelu', w1])
    # insert kernel1 here
    self.pool1 = make_layers(['gelu', 'Md'])
    self.conv2 = make_layers([w2, 'gelu', w2, 'gelu', w2])
    # insert kernel2 here
    self.pool2 = make_layers(['gelu', 'Md'])
    self.conv3 = make_layers(['nopad'])
    # insert kernel3 here
    self.conv4 = make_layers(['gelu', 'NIN'])
    # insert kernel here
    self.conv5 = make_layers(['gelu', 'NIN'])
    # insert kernel here
    self.pool5 = make_layers(['gelu', 'A'])
    
    self.classifier = nn.Linear(self.width2, num_classes)
    

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        
  def get_new_kernels(self, epoch_count):
    if epoch_count % self.epoch == 0 and epoch_count is not 0:
      self.std *= 0.9

    self.kernel1 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=self.width1)
    self.kernel2 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=self.width2)
    self.kernel3 = get_gaussian_filter(kernel_size=3, sigma=self.std, channels=self.width2)
    self.kernel4 = get_gaussian_filter(kernel_size=1, sigma=self.std, channels=self.width2)
    self.kernel5 = get_gaussian_filter(kernel_size=1, sigma=self.std, channels=self.width2)
    

  def forward(self, x):
    # x = self.features(x)
    # x = x.view(x.size(0), -1)
    # x = self.classifier(x)
    
    x = self.conv1(x)
    x = self.kernel1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.kernel2(x)
    x = self.pool2(x)
    
    x = self.conv3(x)
    x = self.kernel3(x)
    
    x = self.conv4(x)
    x = self.kernel4(x)
    
    x = self.conv5(x)
    x = self.kernel5(x)
    x = self.pool5(x)
    
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
  
    return x