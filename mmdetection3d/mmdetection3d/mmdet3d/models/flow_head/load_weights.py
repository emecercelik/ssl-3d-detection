# from __future__ import absolute_import
import torch
import os
import sys

# sys.path.append(os.getcwd())

from flownet3d_head import FlowNet3D

path = 'net_head_tf_nofp.pth'
model = FlowNet3D()
pretrained_weights = torch.load(path)
model_dict = model.state_dict()
state_dict = {k : v for k, v in pretrained_weights.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.load_state_dict(model_dict)
