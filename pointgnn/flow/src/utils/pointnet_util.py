""" PointNet++ Layers

Original Author: Charles R. Qi
Modified by Xingyu Liu
Date: April 2019
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from flow.src.tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from flow.src.tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util


### Sampling and grouping, returns new xyz, new points/features, indices from ball point query, grouped xyz
def sample_and_group(npoint, radius, nsample, xyz, points, knn=False,
                     use_xyz=True):
    '''
    Input:
        npoint: int32 = 1024
        radius: float32 = 0.5,1,2,4
        nsample: int32 = 16
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    new_xyz = gather_point(xyz, farthest_point_sample(npoint,
                                                      xyz))  ### Sampling using farthest point sampling
    # import ipdb; ipdb.set_trace()
    print ('check for seg fault')

    # xyz.shape
    # TensorShape([Dimension(4), Dimension(2048), Dimension(3)])
    # new_xyz.shape
    # TensorShape([Dimension(4), Dimension(1024), Dimension(3)])
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)

    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz,
                                        new_xyz)  ### Grouping using ball query
    grouped_xyz = group_point(xyz,
                              idx)  # (batch_size, npoint, nsample, 3)  ### Resulting grouped coordinates
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2),
                           [1, 1, nsample, 1])  ### translation normalization
    if points is not None:
        grouped_points = group_point(points,
                                     idx)  # (batch_size, npoint, nsample, channel)   ### Resulting grouped features
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points],
                                   axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(
        np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
        dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)),
                              (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (
    batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points],
                                   axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

def get_interpolated_rgb(xyz, point_cloud):

    _, idx = knn_point(3, point_cloud[:, :, :3], xyz)
    grouped_xyz_rgb = group_point(point_cloud, idx) # (batch size, 2048, 3, 6)

    dist = ((tf.expand_dims(xyz, 2) - grouped_xyz_rgb[:, :, :, :3])**2)
    dist = tf.maximum(dist, 1e-10)

    norm = tf.reduce_sum((1/dist), axis=2) # (batch, 2048, 3)

    weighted_rgb = (1/dist)*grouped_xyz_rgb[:, :, :, 3:]
    weighted_norm_rgb = tf.reduce_sum(weighted_rgb, 2)/norm

    return weighted_norm_rgb, dist, grouped_xyz_rgb

# Point Net Layer, returns new xyz, new features, indices from ball point query
def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2,
                       group_all, is_training, bn_decay, scope, bn=True,
                       pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor    (16, 2048, 3)
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        # group_all is passed as False
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz,
                                                                         points,
                                                                         use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint,
                                                                     radius,
                                                                     nsample,
                                                                     xyz,
                                                                     points,
                                                                     knn,
                                                                     use_xyz)

        ### POINT NET LAYER

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])

        # MLP
        for i, num_out_channel in enumerate(mlp):
            # mlp = 3; tf_util.conv2d is done executed 3 times
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)

        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True,
                                       name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True,
                                        name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True,
                                       name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True,
                                        name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d' % (i),
                                            bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points,
                                [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list,
                           mlp_list, is_training, bn_decay, scope, bn=True,
                           use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2),
                                   [1, 1, nsample, 1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz],
                                               axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points,
                                                       [0, 3, 1, 2])
            for j, num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel,
                                                [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=bn, is_training=is_training,
                                                scope='conv%d_%d' % (i, j),
                                                bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points,
                                                       [0, 2, 3, 1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay,
                       scope, bn=True, last_mlp_activation=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        ### INTERPOLATION
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points,
                                                    points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i),
                                         bn_decay=bn_decay,
                                         activation_fn=activation_fn)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1

## our flow embedding layer

def flow_embedding_module_all(xyz1, xyz2, feat1, feat2, radius, nsample, mlp,
                          is_training, bn_decay, scope, bn=True, pooling='max',
                          knn=True, corr_func='elementwise_product'):
    """
    Input:
        xyz1: (batch_size, npoint, 3)
        xyz2: (batch_size, npoint, 3)
        feat1: (batch_size, npoint, channel)
        feat2: (batch_size, npoint, channel)
    Output:
        xyz1: (batch_size, npoint, 3)
        feat1_new: (batch_size, npoint, mlp[-1])
    """

    feat1_expanded = tf.expand_dims(feat1, 2)  # batch_size, npoint, 1, 3
    feat2_expanded = tf.expand_dims(feat2, 1)

    xyz1_expanded = tf.expand_dims(xyz1, 2)  # batch_size, npoint, 1, 3
    xyz2_expanded = tf.expand_dims(xyz2, 1)

    xyz_diff = xyz2_expanded - xyz1_expanded  # batch_size, npoint, nsample, 3    ### Distance vector

    # (16, 256, 256, 256)
    feat_diff = tf.concat(axis=-1, values=[tf.tile(feat2_expanded, [1, nsample, 1, 1]),
                                           tf.tile(feat1_expanded, [1, 1, nsample, 1])])

    # Concat the coordinates and features difference here
    feat1_new = tf.concat([feat_diff, xyz_diff], axis=3)  # batch_size, npoint, nsample, [channel or 1] + 3

    # TODO: move scope to outer indent
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=True, is_training=is_training,
                                       scope='conv_diff_%d' % (i),
                                       bn_decay=bn_decay)

    # Tensor("flow_embedding/conv_diff_0/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)
    # Tensor("flow_embedding/conv_diff_1/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)
    # Tensor("flow_embedding/conv_diff_2/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)

    if pooling == 'max':
        feat1_new = tf.reduce_max(feat1_new, axis=[2], keep_dims=False,
                                  name='maxpool_diff')
    elif pooling == 'avg':
        feat1_new = tf.reduce_mean(feat1_new, axis=[2], keep_dims=False,
                                   name='avgpool_diff')
    return xyz1, feat1_new



########


def flow_embedding_module(xyz1, xyz2, feat1, feat2, radius, nsample, mlp,
                          is_training, bn_decay, scope, bn=True, pooling='max',
                          knn=True, corr_func='elementwise_product'):
    """
    Input:
        xyz1: (batch_size, npoint, 3)
        xyz2: (batch_size, npoint, 3)
        feat1: (batch_size, npoint, channel) (16, 256, 128)
        feat2: (batch_size, npoint, channel) (16, 256, 128) 
    Output:
        xyz1: (batch_size, npoint, 3)
        feat1_new: (batch_size, npoint, mlp[-1])
    """
    """
    our case 16 x 256 x 256 x (128 + 128)
    """
    if knn:
        _, idx = knn_point(nsample, xyz2, xyz1)
        print('knn')
    else:
        idx, _ = query_ball_point(radius, nsample, xyz2, xyz1)
        print('query')
    # finds the indices got from knn in pc2 xyz


    xyz2_grouped = group_point(xyz2,
                               idx)  # batch_size, npoint, nsample, 3      ### Point cloud 2
    xyz1_expanded = tf.expand_dims(xyz1,
                                   2)  # batch_size, npoint, 1, 3          ### Point cloud 1
    xyz_diff = xyz2_grouped - xyz1_expanded  # batch_size, npoint, nsample, 3    ### Distance vector

    feat2_grouped = group_point(feat2,
                                idx)  # batch_size, npoint, nsample, channel
    feat1_expanded = tf.expand_dims(feat1, 2)  # batch_size, npoint, 1, channel
    # TODO: change distance function
    if corr_func == 'elementwise_product':
        feat_diff = feat2_grouped * feat1_expanded  # batch_size, npoint, nsample, channel
    elif corr_func == 'concat':  ### Using this one
        feat_diff = tf.concat(axis=-1, values=[feat2_grouped,
                                               tf.tile(feat1_expanded,
                                                       [1, 1, nsample,
                                                        1])])  # batch_size, npoint, sample, channel*2
    elif corr_func == 'dot_product':
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1],
                                  keep_dims=True)  # batch_size, npoint, nsample, 1
    elif corr_func == 'cosine_dist':
        feat2_grouped = tf.nn.l2_normalize(feat2_grouped, -1)
        feat1_expanded = tf.nn.l2_normalize(feat1_expanded, -1)
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1],
                                  keep_dims=True)  # batch_size, npoint, nsample, 1
    elif corr_func == 'flownet_like':  # assuming square patch size k = 0 as the FlowNet paper
        batch_size = xyz1.get_shape()[0].value
        npoint = xyz1.get_shape()[1].value
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1],
                                  keep_dims=True)  # batch_size, npoint, nsample, 1
        total_diff = tf.concat(axis=-1, values=[xyz_diff,
                                                feat_diff])  # batch_size, npoint, nsample, 4
        feat1_new = tf.reshape(total_diff, [batch_size, npoint,
                                            -1])  # batch_size, npoint, nsample*4
        # feat1_new = tf.concat(axis=[-1], values=[feat1_new, feat1]) # batch_size, npoint, nsample*4+channel
        return xyz1, feat1_new

    # Concat the coordinates and features difference here
    feat1_new = tf.concat([feat_diff, xyz_diff],
                          axis=3)  # batch_size, npoint, nsample, [channel or 1] + 3
    # TODO: move scope to outer indent
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=True, is_training=is_training,
                                       scope='conv_diff_%d' % (i),
                                       bn_decay=bn_decay)

    # Tensor("flow_embedding/conv_diff_0/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)
    # Tensor("flow_embedding/conv_diff_1/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)
    # Tensor("flow_embedding/conv_diff_2/Relu:0", shape=(16, 256, 64, 128), dtype=float32, device= / device: GPU:0)

    if pooling == 'max':
        feat1_new = tf.reduce_max(feat1_new, axis=[2], keep_dims=False,
                                  name='maxpool_diff')
    elif pooling == 'avg':
        feat1_new = tf.reduce_mean(feat1_new, axis=[2], keep_dims=False,
                                   name='avgpool_diff')
    return xyz1, feat1_new


def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training,
                      scope, bn_decay=None, bn=True, pooling='max', radius=None,
                      knn=True):
    """
        Feature propagation from xyz2 (less points) to xyz1 (more points)

    Inputs:
        xyz1: (batch_size, npoint1, 3)
        xyz2: (batch_size, npoint2, 3)
        feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
        feat2: (batch_size, npoint2, channel2) features for xyz2 points
    Output:
        feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

        TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
    """
    with tf.variable_scope(scope) as sc:
        if knn:
            l2_dist, idx = knn_point(nsample, xyz2, xyz1)
        else:
            idx, pts_cnt = query_ball_point(radius, nsample, xyz2, xyz1)

        xyz2_grouped = group_point(xyz2, idx)  # batch_size, npoint1, nsample, 3
        xyz1_expanded = tf.expand_dims(xyz1, 2)  # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded  # batch_size, npoint1, nsample, 3

        feat2_grouped = group_point(feat2,
                                    idx)  # batch_size, npoint1, nsample, channel2
        net = tf.concat([feat2_grouped, xyz_diff],
                        axis=3)  # batch_size, npoint1, nsample, channel2+3

        if mlp is None: mlp = []
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d' % (i), bn_decay=bn_decay)
        if pooling == 'max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False,
                                      name='maxpool')  # batch_size, npoint1, mlp[-1]
        elif pooling == 'avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False,
                                       name='avgpool')  # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1],
                                  axis=2)  # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new,
                                   2)  # batch_size, npoint1, 1, mlp[-1]+channel2
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d' % (i),
                                       bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2])  # batch_size, npoint1, mlp2[-1]
        return feat1_new
