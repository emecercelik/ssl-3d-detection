from torch import nn as nn

from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index

class Sample(nn.Module):
    def __init__(self, num_points):
        super(Sample, self).__init__()
        
        self.num_points = num_points
        
    def forward(self, points):
        new_points_ind = furthest_point_sampling(points.permute(0, 2, 1).contiguous(), self.num_points)
        new_points = fps_gather_by_index(points.contiguous(), new_points_ind)
        return new_points