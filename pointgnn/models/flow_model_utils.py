"""This file implements functions to encode and decode 3boxes."""

import numpy as np
import tensorflow as tf
from flow.src.tf_ops.sampling.tf_sampling import farthest_point_sample,gather_point
import IPython

def get_flow_placeholders(config, box_encoding_len):
    '''
    To get placeholders for the second point cloud aligned with the first 
    point cloud used for GNN. 

    Parameters
    ----------
    config : Json configuration file. 
        Config of Point-GNN
    box_encoding_len: Scalar
        Defines with how many parameters the 3D box will be encoded

    Returns
    -------
    t_initial_vertex_features: float32 Placeholder [None,?], ? determined with
        the feature type (only reflectance, or with rgb,...)
    t_vertex_coord_list: float32 Placeholder [[None,3],[None,3],[None,3]]
    t_edges_list : int32 Placeholder [[None,None], [None,None]]
        In each graph level defined by the total num of points and num of 
        keypoints
    t_keypoint_indices_list: int32 Placeholder [[None,1], [None,1]]
    t_class_labels: int32 Placeholder [None,1]
    t_encoded_gt_boxes: float32 Placeholder [None,1,box_encoding_len]
    t_valid_gt_boxes: float32 Placeholder [None,1,1]

    '''
    if config['flow_parameters']['train'] == 'flow' or config['flow_parameters']['train'] == 'both':
        if config['input_features'] == 'irgb':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 4],name='ph_pc2_t_initial_vertex_features')
        elif config['input_features'] == 'rgb':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 3],name='ph_pc2_t_initial_vertex_features')
        elif config['input_features'] == '0000':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 4],name='ph_pc2_t_initial_vertex_features')
        elif config['input_features'] == 'i000':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 4],name='ph_pc2_t_initial_vertex_features')
        elif config['input_features'] == 'i':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 1],name='ph_pc2_t_initial_vertex_features')
        elif config['input_features'] == '0':
            t_initial_vertex_features = tf.placeholder(
                dtype=tf.float32, shape=[None, 1],name='ph_pc2_t_initial_vertex_features')
    
        t_vertex_coord_list = [
            tf.placeholder(dtype=tf.float32, shape=[None, 3],name='ph_pc2_t_vertex_coord_list0')]
        for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
            t_vertex_coord_list.append(
                tf.placeholder(dtype=tf.float32, shape=[None, 3],name='ph_pc2_t_vertex_coord_list{}'.format(_+1)))
    
        t_edges_list = []
        for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
            t_edges_list.append(
                tf.placeholder(dtype=tf.int32, shape=[None, None],name='ph_pc2_t_edges_list{}'.format(_)))
    
        t_keypoint_indices_list = []
        for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
            t_keypoint_indices_list.append(
                tf.placeholder(dtype=tf.int32, shape=[None, 1],name='ph_pc2_t_keypoint_indices_list{}'.format(_)))
    
        t_class_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1],name='ph_pc2_t_class_labels')
        t_encoded_gt_boxes = tf.placeholder(
            dtype=tf.float32, shape=[None, 1, box_encoding_len],name='ph_pc2_t_encoded_gt_boxes')
        t_valid_gt_boxes = tf.placeholder(
            dtype=tf.float32, shape=[None, 1, 1],name='ph_pc2_t_valid_gt_boxes')
        
        t_n_keypoints = tf.placeholder(dtype=tf.int32, shape=[None],name='n_keypoints_pc2')
        t_n_points = tf.placeholder(dtype=tf.int32, shape=[None],name='n_points_pc2')
        
        return t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list, \
            t_edges_list, t_class_labels, t_encoded_gt_boxes, \
            t_valid_gt_boxes, t_n_keypoints, t_n_points
    else:
        return None, None, None, \
            None, None, None, \
            None, None, None
        
def convert_batch_shape(features,xyz,n_batch,batch_npoints,npoint):
    '''
    To split features and xyz of keypoints into batches using the given number
    of points in each batch. The number of keypoints is fixed for each frame
    of the batch. 

    Parameters
    ----------
    features : Tensor with (n,m) shape.
        Features of the keypoints. n is the number of keypoints and m is the
        number of features. 
    xyz : Tensor with (n,3) shape.
        xyz values of keypoints. n is the number of keypoints.
    n_batch : scalar
        Number of frames in the batch.
    batch_npoints : Tensor with [None] shape
        A list tensor that takes number of points in each frame of batch. 
    npoint : Scalar
        Number of points that will be sampled from each frame. If n<npoint, 
        points are sampled with replacement to have npoints in total.
        Farthest point sampling is used for sampling. 

    Returns
    -------
    feature_batch_tensors : Tensor with (n_batch,npoint,m)
        Features of the keypoints of frames in the batch.
    xyz_batch_tensors : Tensor with (n_batch,npoint,3)
        Coordinates of the keypoints of frames in the batch.

    '''
    counter = 0
    feature_batch_tensors = []
    xyz_batch_tensors = []
    for i in range(n_batch):
        single_feat = tf.expand_dims(features[counter:counter+batch_npoints[i],:],0)
        single_xyz = tf.expand_dims(xyz[counter:counter+batch_npoints[i],:],0)
        sampled_feat = tf.gather(single_feat,farthest_point_sample(npoint,single_xyz[:,:,0:3])[0],
                                 axis=1)
        sampled_xyz = tf.gather(single_xyz,farthest_point_sample(npoint,single_xyz[:,:,0:3])[0],
                                 axis=1)
        
        #feat = gather_point(single_feat,)
        
        feature_batch_tensors.append(sampled_feat)
        xyz_batch_tensors.append(sampled_xyz)
        
        counter+=batch_npoints[i]
    feature_batch_tensors = tf.stack(feature_batch_tensors,axis=0)[:,0,:,:]
    xyz_batch_tensors = tf.stack(xyz_batch_tensors,axis=0)[:,0,:,:]
    return feature_batch_tensors,xyz_batch_tensors

def test_convert_batch_shape():
    feat_inp = tf.placeholder(dtype=tf.float32,shape=[None,10])
    batch_sizes = tf.placeholder(dtype=tf.int32,shape=[None])
    batch = [2,11,13,36]
    n_batch = len(batch)
    npoint = 5
    features = np.random.rand(np.sum(batch),10)
    split_features, split_xyz = convert_batch_shape(feat_inp,feat_inp,n_batch,batch_sizes,npoint)
    sess = tf.Session()
    outp = sess.run(split_features,feed_dict={feat_inp:features,batch_sizes:batch})
    for i,o in enumerate(outp):
        st = int(np.sum(batch[:i]))
        end = int(np.sum(batch[:i+1]))
        for vec in o:
            try:
                assert np.isclose(np.sum(features[st:end] - vec,axis=1),0,atol=1e-6).any()
                print('Successful!')
            except:
                IPython.embed()
                        
def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.sum(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.sum(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.sum(EPE)
    return EPE, acc1, acc2, error, gtflow_len   
    
if __name__=='__main__':   
    test_convert_batch_shape()
    
    
    
