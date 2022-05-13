# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from .group import Group
from .sample import Sample

class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()
        
        self.sample = Sample(num_points)
        self.group = Group(radius, num_samples)
        
        layers = []
        out_channels = [in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points, features):
        new_points = self.sample(points)
        new_features = self.group(points, new_points, features)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_points, new_features
