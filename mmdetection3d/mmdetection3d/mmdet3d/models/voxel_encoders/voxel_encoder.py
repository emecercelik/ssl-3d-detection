import torch
import os
import time
import numpy as np
import pickle
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn
from torch_geometric.nn import knn
from mmdet3d.models import voxel_encoders

from mmdet3d.ops import DynamicScatter
from .. import builder
from ..builder import VOXEL_ENCODERS
from ..builder import build_loss
from ..builder import build_head
from .utils import VFELayer, get_paddings_indicator

from mmdet3d.models.flow_head.flownet3d_head import FlowNet3D

store_path = '/mmdetection3d/saved_points'

@VOXEL_ENCODERS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int): Number of features to use. Default: 4.
    """

    def __init__(self, num_features=4):
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
        """
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()


@VOXEL_ENCODERS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@VOXEL_ENCODERS.register_module()
class DynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(DynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class HardVFE(nn.Module):
    """Voxel feature encoder used in DV-SECOND.
    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 flow = False,
                 f_cycle_loss=dict(type='Cycle_Loss', loss_weight=1),
                 flownet = dict(type='FlowNet3D')):
        super(HardVFE, self).__init__()
        # initial cycle_loss
        self.flow = flow
        self.f_cycle_loss = build_loss(f_cycle_loss)
        # initial Flownet3D
        self.flownet = build_head(flownet)
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                num_points,
                coors,
                img_feats=None,
                img_metas=None):
        """Forward functions.
        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.
        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
       
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        
        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)
        features_org = voxel_feats # N * 64 *10
         #(N, 32, 10) 
        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
     
        if self.flow:
        # store the 2048 point features with coors
            dtype = voxel_feats.dtype
            # features = features.view(-1, 9)
            batch_size = 2
            for batch_itt in range(batch_size-1):
                # Distinguish first sample and second sample according to the first column of tensor
                first_batch_mask = coors[:, 0] == batch_itt
                second_batch_mask = coors[:, 0] != batch_itt  # There is only one sample input in the test, and an empty set may appear
                
            # Pillars
                # get the coors of pillars in the first sample and the second sample
                # first_frame_pillar_coors = coors[first_batch_mask, :][:, :3] # (f1, 3), 4 is (sample_num, x, y, z) (f1 is the nnumber of voxel in the first frame)
                # second_frame_pillar_coors = coors[second_batch_mask, :][:, :3] # (f2, 3)
                
                # get the features of pillars in the first sample and the second sample
                first_frame_pillar_features = voxel_feats[first_batch_mask, :] # (f1, 64, 64) # notice: there should be feaatures after VEF or PFNs
                second_frame_pillar_features = voxel_feats[second_batch_mask, :] # (f2, 64, 64) 
                
        # Step 1  Input P1 to KNN and find 1NN of P1 (named P2)
        # first_sample_point is P1, second_sample_point is P2
            # Points
                # get points features in each sample
                first_frame_point_features = first_frame_pillar_features.view(-1, 64) # (f1*64, 64)
                second_frame_point_features = second_frame_pillar_features.view(-1, 64) # (f2*64, 64)
                # print("first_frame_point_features {}".format(first_frame_pillar_features))
                # get points coors in each sample
                first_frame_point_coors = features_org[first_batch_mask, :].view(-1, 10) # (f1*64, 10)
                second_frame_point_coors = features_org[second_batch_mask, :].view(-1, 10) # (f2*64, 10)
                first_frame_point_coors = first_frame_point_coors[:, :3].contiguous() # (f1*64, 3)
                second_frame_point_coors = second_frame_point_coors[:, :3].contiguous() # (f2*64, 3)
        
            # concatenate coors and features
                first_frame_point = torch.cat((first_frame_point_coors, first_frame_point_features), dim=1)
                second_frame_point = torch.cat((second_frame_point_coors, second_frame_point_features), dim=1)
            
            # get indecies of non-empty points 
                non = torch.zeros(1, 3).cuda(0)
                non_index1 = first_frame_point_coors != non
                non_index2 = second_frame_point_coors != non
            
                first_frame_point_ne = first_frame_point[non_index1[:, 0]]
                second_frame_point_ne = second_frame_point[non_index2[:, 0]]
                # print(first_frame_point_ne.shape)
            # get non-empty points coors 
                first_frame_point_coors_ne = first_frame_point_ne[:, :3]
                second_frame_point_coors_ne = second_frame_point_ne[:, :3]
            # get non-empty points featrues
                first_frame_point_features_ne = first_frame_point_ne[:, 3:]
                second_frame_point_features_ne = second_frame_point_ne[:, 3:]
                # print(first_frame_point_features)
                # print('first_smaple {}'.format(first_frame_point_features_ne))
            # Sample points
                # random sample 2048 points from the first sample
                #first_indices = torch.randperm(len(first_frame_point_coors_ne))[:2048] # sample 4096 points instead of 2048
                first_seed = torch.randint(0,int((len(first_frame_point_coors_ne)-2049)), (1,))
                first_sample_point_coors = first_frame_point_coors_ne[first_seed : (first_seed + 2048)]#.contiguous() # (2048, 3) 
                first_sample_point_features = first_frame_point_features_ne[first_seed : (first_seed + 2048)] # (2048, 64) 
                
                # use KNN in the second frame to get 2 neighbor points of the random sampled two points in the first sample
                # second raw is the indices of neighbor points 
                edge = knn(second_frame_point_coors_ne.to(dtype), first_sample_point_coors.to(dtype), 1) # (2, 2048)
                second_indices = edge[1:,].view(2048)
                # second_indices = torch.randperm(len(second_frame_point_coors_ne))[:2048]

                # get the coors and features of sample points in second frame
                second_sample_point_coors = second_frame_point_coors_ne[second_indices] # (2048, 3)
                second_sample_point_features = second_frame_point_features_ne[second_indices] # (2048, 64)
                
        # Step 2 Use P1 and P2 as input of Flownet to get Flow1
            # 将flownet作为类写在一个文件里，在PillarFeaturesNet中调用flownet，flownet返回features_pred
            # flownet = FlowNet3D().cuda()
            first_sample_point_coors = first_sample_point_coors.unsqueeze(dim=0).permute(0, 2, 1).contiguous() # (1, 3, 2048)
            first_sample_point_features = first_sample_point_features.unsqueeze(dim=0).permute(0, 2, 1).contiguous() # (1, 64, 2048)
            
            second_sample_point_coors = second_sample_point_coors.unsqueeze(dim=0).permute(0, 2, 1).contiguous()
            # print("second sample1 {}".format(second_sample_point_coors.shape))
            # second_sample_point_features_org = second_sample_point_features
            second_sample_point_features = second_sample_point_features.unsqueeze(dim=0).permute(0, 2, 1).contiguous()

            # forward_prediction
            # features1 = flownet(first_frame_point_coors, first_frame_point_features, first_sample_point_coors, second_sample_point_coors, first_sample_point_features, second_sample_point_features) (f1*32, 64)
            features_pred_flow1 = self.flownet(first_sample_point_coors, second_sample_point_coors, first_sample_point_features, second_sample_point_features)
            
            
        # Step 3 Flow1 + P1 to get P2_hat(pred_f)
            pred_f = first_sample_point_coors + features_pred_flow1 # (1, 3, N)
            pred_f_new = pred_f.squeeze().permute(1, 0) # (N, 3)
            
            # print("pred_f_new {}".format(pred_f_new))
            
        # Step 4 Use 1NN again to get 1NN of P2_hat (named P2_hatNN) in P2
            # nprint(second_sample_point_coors.shape)
            second_sample_point_coors1 = second_sample_point_coors.squeeze().permute(1, 0)
            edge_pdf = knn(second_sample_point_coors1.to(dtype), pred_f_new.to(dtype), 1)
            # edge_pdf = knn(second_frame_point_coors_ne.to(dtype), pred_f_new.to(dtype), 1) # use original point cloud coors in frame2 as input or ???需要测试
            # print("edge {}".format(edge_pdf.shape))
            p2_hatnn_indicies = edge_pdf[1:,].view(2048)
            # p2_hatnn_coors = second_frame_point_coors_ne[p2_hatnn_indicies] # (N, 3)
            p2_hatnn_coors = second_sample_point_coors1[p2_hatnn_indicies]
            # p2_hatnn_features = second_frame_point_features_ne[p2_hatnn_indicies] #(N, 64)
            
            p2_hatnn_coors = p2_hatnn_coors.unsqueeze(dim=0).permute(0, 2, 1).contiguous()#.view(0, 2, 1)
            # print("second_sample_point_features_org {}".format(second_sample_point_features_org.shape)) # (N, 64)
        # Step 5 Calculate the average of P2_hat and P2_hatNN, named P2_avg
            p2_avg_coors = (pred_f + p2_hatnn_coors) / 2
            
        # Step 6 Use P2_avg and P1 as input to Flownet again to get Flow2
            # we use twice first_sample_point_features
            features_pred_flow2 = self.flownet(p2_avg_coors, first_sample_point_coors, first_sample_point_features, first_sample_point_features) 
            # features_pred_flow2 = flownet(p2_avg_coors, first_sample_point_coors, p2_avg_features, first_sample_point_features)
            
        # Step 7 P2_avg + Flow2 to get P1_hat(pred_b)
            pred_b = p2_avg_coors + features_pred_flow2

            voxel_feats = torch.max(voxel_feats, keepdim=True, dim=1)[0] # (f1, 1, 64)

            return voxel_feats.squeeze(), pred_f, p2_hatnn_coors, pred_b, first_sample_point_coors

 
        else:
            if (self.fusion_layer is not None and img_feats is not None):
                voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                    coors, img_feats, img_metas)
           #  voxel_feats = torch.max(voxel_feats, keepdim=True, dim=1)[0] # (f1, 1, 64)
            return voxel_feats#.squeeze()
    # # use max_pooling to get the output voxel_wise features
        
    
        # ========================== Save point clouds =========================
        # first_frame_point_coors_ne_c = first_frame_point_coors_ne.to('cpu')
        # second_frame_point_coors_ne_c = second_frame_point_coors_ne.to('cpu')
        # first_sample_point_coors_c = first_sample_point_coors.to('cpu')
        # pred_f_c = pred_f.to('cpu')
        # pts_dict = {'first_frame_point': first_frame_point_coors_ne_c, 'second_frame_point': second_frame_point_coors_ne_c, 'first_sample_point_coors': first_sample_point_coors_c , 'predicted_flow': pred_f_c}
        # local_time = time.localtime()
        # file_name = '{}_{}_{}_{}_{}.npy'.format(local_time.tm_mon, local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec)
        # os.system('mkdir -p {}'.format(store_path))
        # file_path = os.path.join(store_path, file_name)
        # # file_path = store_path
        # # file_path = open(file_path, file_name)
        # with open (file_path, 'wb') as f:
        #     pickle.dump(pts_dict, f)
        # ========================== End of save ================================
        
        
    
    def fusion_with_mask(self, features, mask, voxel_feats, coors, img_feats,
                         img_metas):
        """Fuse image and point features with mask.
        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.
        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out
