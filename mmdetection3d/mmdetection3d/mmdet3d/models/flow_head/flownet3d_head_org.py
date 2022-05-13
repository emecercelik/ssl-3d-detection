import torch
from torch import nn as nn
from .pointnet_setconv import SetConv
from .pointnet_setupconv import SetUpConv
from .flow_embedding import FlowEmbedding
from .pointnet_featprop import FeaturePropagation

class FlowNet3D(nn.Module):
    def __init__(self):
        super(FlowNet3D, self).__init__()

        self.set_conv1 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        self.set_conv2 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv3 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        # self.fp = FeaturePropagation(256, 3, [256, 256])
        self.fp = FeaturePropagation(256, 64, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )
         
    def forward(self, points1, points2, features1, features2):
        points1_2, features1_2 = self.set_conv1(points1, features1)
        points2_2, features2_2 = self.set_conv1(points2, features2)

        embedding = self.flow_embedding(points1_2, points2_2, features1_2, features2_2)
        
        points1_3, features1_3 = self.set_conv2(points1_2, embedding)
        points1_4, features1_4 = self.set_conv3(points1_3, features1_3)
        
        new_features1_3 = self.set_upconv1(points1_4, points1_3, features1_4, features1_3)
        new_features1_2 = self.set_upconv2(points1_3, points1_2, new_features1_3, torch.cat([features1_2, embedding], dim=1))

        # new_features1 = self.set_upconv3(points1_2, points1, new_features1_2, features1)
        new_features1 = self.fp(points1_2, points1, new_features1_2, features1)

        flow = self.classifier(new_features1)
        
        return flow
