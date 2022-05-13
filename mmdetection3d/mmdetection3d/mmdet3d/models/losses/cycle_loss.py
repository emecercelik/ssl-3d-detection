# Author: Zhijie Yang
# Email: zhijie.yang@tum.de
# Source: https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..builder import LOSSES
from mmdet.models.losses.utils  import weighted_loss

writer = SummaryWriter('runs/cycle_loss')

# @weighted_loss
def cycle_loss(pred_f, 
               grouped_xyz, 
               pred_b, 
               point_cloud1, 
               end_points=None,
               rigidity=False, 
               rgb=False, 
               point_cloud1_rgb=None, 
               flip_prefix='', 
               cycle_loss_weight=1,
               knn_loss_weight=1):
        
        end_points_loss = {}

        # knn_l2_loss = knn_loss_weight * (1 / pred_f.shape[2]) * \
        #     (torch.sum((pred_f - grouped_xyz) * (pred_f - grouped_xyz), axis=2)).norm() / 2.0
        knn_l2_loss = knn_loss_weight * torch.mean(
        torch.sum((pred_f - grouped_xyz) * (pred_f - grouped_xyz), axis=2)) / 2.0
        # knn_l2_loss = knn_loss.norm()
       
        writer.add_scalar('{}KNN L2 loss'.format(flip_prefix), knn_l2_loss)

        end_points_loss['knn_l2_loss'] = knn_l2_loss

        cycle_l2_loss = cycle_loss_weight * torch.mean(
        torch.sum((pred_b - point_cloud1) * (pred_b - point_cloud1), axis=2)) / 2.0
        # cycle_l2_loss = cycle_loss_weight * (1 / pred_b.shape[2]) * \
        #     (torch.sum((pred_b - point_cloud1) * (pred_b - point_cloud1), axis=2)).norm() / 2.0
        # cycle_l2_loss = cycle_loss.norm()
        writer.add_scalar('{}Cycle l2 loss'.format(flip_prefix), cycle_l2_loss)

        end_points_loss['cycle_l2_loss'] = cycle_l2_loss

        l2_loss = knn_l2_loss + cycle_l2_loss

        end_points_loss['l2_loss'] = l2_loss
        writer.add_scalar('{}Total l2 loss'.format(flip_prefix), l2_loss)

        return l2_loss#, end_points_loss # In fact we just use l2_loss


@LOSSES.register_module()
class Cycle_Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(Cycle_Loss, self).__init__()
        self.cycle_loss_weight = loss_weight
    
    def forward(self,
                pred_f, 
                grouped_xyz, 
                pred_b, 
                point_cloud1, 
                end_points=None,
                rigidity=False, 
                rgb=False, 
                point_cloud1_rgb=None, 
                flip_prefix='', 
                cycle_loss_weight=1,
                knn_loss_weight=1):

        
        l2_loss = cycle_loss(pred_f, grouped_xyz, pred_b, point_cloud1)
        return l2_loss