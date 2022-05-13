import numpy as np
import torch
from torch import nn as nn
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import group_gather_by_index

def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

class Group(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()
        
        self.radius = radius
        self.num_samples = num_samples
        self.knn = knn
        
    def forward(self, points, new_points, features):
        if self.knn:
            dist = pdist2squared(points, new_points)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, points.permute(0, 2, 1).contiguous(),
                             new_points.permute(0, 2, 1).contiguous(), False)
        grouped_points = group_gather_by_index(points, ind)
        grouped_points -= new_points.unsqueeze(3)
        if features is None:
            return grouped_points, ind
        grouped_features = group_gather_by_index(features, ind)
        new_features = torch.cat([grouped_points, grouped_features], dim=1)
        return new_features

