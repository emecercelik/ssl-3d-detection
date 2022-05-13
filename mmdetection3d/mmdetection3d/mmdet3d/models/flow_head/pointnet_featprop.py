import torch
from torch import nn as nn

from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()
        
        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points1, points2, features1, features2):
        dist, ind = three_nn(points2.permute(0, 2, 1).contiguous(), points1.permute(0, 2, 1).contiguous())
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        #new_features = three_interpolate(features1, ind, weights) # wrong gradients
        new_features = torch.sum(group_gather_by_index(features1, ind) * weights.unsqueeze(1), dim = 3)
        new_features = torch.cat([new_features, features2], dim=1)
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        return new_features
