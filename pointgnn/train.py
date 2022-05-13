"""This file defines the training process of Point-GNN object detection."""

import os
import time
import argparse
import copy
from sys import getsizeof
from multiprocessing import Pool, Queue, Process

import numpy as np
import tensorflow as tf

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn,\
    get_encoding_len
from models.crop_aug import CropAugSampler
from models import preprocess
from models.flow_model_utils import get_flow_placeholders,scene_flow_EPE_np
from util.tf_util import average_gradients
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from util.summary_util import write_summary_scale
from collections import namedtuple, defaultdict
from models.var_list import detection_head, scene_flow_head, backbone
from shutil import copyfile
import IPython

def read_min_loss(path):
    if os.path.isfile(path):
        with open(path,'r') as f:
            return float(f.readlines()[0])
    else:
        return 1000.
       
       
def write_min_loss(path,loss):
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path,'w') as f:
        f.write(str(loss))
        

#os.environ['TF_CUDNN_DETERMINISTIC']='1'
parser = argparse.ArgumentParser(description='Training of PointGNN')
parser.add_argument('train_config_path', type=str,
                   help='Path to train_config')
parser.add_argument('config_path', type=str,
                   help='Path to config')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits'
                   '/train_config["train_dataset"]"')

args = parser.parse_args()
train_config = load_train_config(args.train_config_path)
DATASET_DIR = args.dataset_root_dir
if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
        './3DOP_splits/'+train_config['train_dataset'])
else:
    DATASET_SPLIT_FILE = args.dataset_split_file
config_complete = load_config(args.config_path)
if 'train' in config_complete:
    config = config_complete['train']
else:
    config = config_complete


if "flow_parameters" not in config:
    config["flow_parameters"] =dict()

if 'supervised' not in config['flow_parameters']:
    config['flow_parameters']['supervised']=False

if "eval_dataset" in config["flow_parameters"]:
    if config["flow_parameters"]["eval_dataset"] is not None:
        FLOW_VAL_DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
        './3DOP_splits/'+config["flow_parameters"]["eval_dataset"])
    else:
        FLOW_VAL_DATASET_SPLIT_FILE = None


# Weight of the flow loss added to the total_loss 
if "flow_loss_weight" in config['flow_parameters']:
    FLOW_LOSS_WEIGHT=config['flow_parameters']['flow_loss_weight']
else:
    FLOW_LOSS_WEIGHT = 0.0

if "train" in config['flow_parameters']:
    TRAIN_OPT=config['flow_parameters']['train'] # To decide flow or 3D det to train: 'flow', '3Ddet', or 'both'
else:
    TRAIN_OPT="3Ddet"
if 'restore_path' in config['flow_parameters']:
    FLOW_RESTORE = config['flow_parameters']['restore_path']
else:
    FLOW_RESTORE = None

# Parameter to sample more than 1 point set in a batch for scene flow training
# Multiplies the batch size 
if 'repeat_batch' in config['flow_parameters']:
    REPEATED_BATCH_FLOW = config['flow_parameters']['repeat_batch']
else:
    REPEATED_BATCH_FLOW = 1    


DET_WEIGHT = 1.
if TRAIN_OPT == '3Ddet':
    FLOW_LOSS_WEIGHT=0.
if TRAIN_OPT == 'flow':
    DET_WEIGHT = 0. # not to use 3D det loss

# input function ==============================================================
dataset = KittiDataset(
    os.path.join(DATASET_DIR, 'image/training/image_2'),
    os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
    os.path.join(DATASET_DIR, 'calib/training/calib/'),
    os.path.join(DATASET_DIR, 'labels/training/label_2'),
    DATASET_SPLIT_FILE,
    num_classes=config['num_classes'])

# Flow self-supervised val dataset
if FLOW_VAL_DATASET_SPLIT_FILE is not None and not config['flow_parameters']['supervised']:
    dataset_val = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        FLOW_VAL_DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])
    if 'validate_every_epoch' not in train_config:
        train_config['validate_every_epoch']=train_config.get('save_every_epoch',1)
else:
    dataset_val = None
# background, horizontal anchor, vertical anchor, donotcare
# Car: 4, Pedestrian and Cyclist: 6
NUM_CLASSES = dataset.num_classes

if 'NUM_TEST_SAMPLE' not in train_config:
    NUM_TEST_SAMPLE = dataset.num_files
else:
    if train_config['NUM_TEST_SAMPLE'] < 0:
        NUM_TEST_SAMPLE = dataset.num_files
        print('Number of training frames: ', dataset.num_files)
    else:
        NUM_TEST_SAMPLE = train_config['NUM_TEST_SAMPLE']

BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

aug_fn = preprocess.get_data_aug(train_config['data_aug_configs'])

if 'crop_aug' in train_config:
    sampler = CropAugSampler(train_config['crop_aug']['crop_filename'])

def fetch_data(frame_idx,dataset):
    # Get camera points that are visible in image and append image color
    #    to the points as attributes.
    #Points['xyz'],Points['attr']: This is a concat of r and rgb
    if not config['flow_parameters'].get('is_nuscenes',False):
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
            config['downsample_by_voxel_size'])  # Points(xyz,[reflections,rgb])
    else:
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb_nuscenes(frame_idx,
            config['downsample_by_voxel_size'])  # Points(xyz,[reflections,rgb])
    box_label_list = dataset.get_label(frame_idx) # a list of label dictionaries
    # Crop Augmentation 
    if 'crop_aug' in train_config:
        cam_rgb_points, box_label_list = sampler.crop_aug(cam_rgb_points,
            box_label_list,
            sample_rate=train_config['crop_aug']['sample_rate'],
            parser_kwargs=train_config['crop_aug']['parser_kwargs'])
    # Other augmentations applied to points
    # TODO: Get the random augmentation values to apply the same to the previous frame
    cam_rgb_points, box_label_list = aug_fn(cam_rgb_points, box_label_list)
    
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    # vertex_coord_list: [points, downsampled_level_1, downsampled_level_2,...]
    # keypoint_indices_list: [indices_downsampled_level_1_in_points, indices_downsampled_level_2_in_points,...]
    # edges_list: [[list_keypoint_level_1_idx_i, list_points_neighbor_idx_j],[list_keypoint_level_2_idx_i,list_keypoint_level_1_idx_j],...]
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])
    #IPython.embed()  
    # b= dataset.cam_points_to_velo(cam_rgb_points,dataset.get_calib(frame_idx))  
    # a = np.hstack([vertex_coord_list[1],vertex_coord_list[1][:,0:1]]);a.tofile('/point_gnn/nuscenes_data_check/{:06d}.bin'.format(frame_idx));print(frame_idx)
    # b= dataset.cam_points_to_velo(cam_rgb_points,dataset.get_calib(frame_idx))  ; a = np.hstack([b.xyz,b.attr[:,0:1]]);a=a.astype(np.float32);a.tofile('/point_gnn/nuscenes_data_check/{:06d}.bin'.format(frame_idx));print(frame_idx);print(len(vertex_coord_list[0]),len(vertex_coord_list[1]))
    # v = vertex_coord_list[0]; k=keypoint_indices_list[0]; p = Points(xyz=v[k[:,0]],attr=np.hstack([v[k[:,0]],v[k[:,0]][:,0:1]]));b= dataset.cam_points_to_velo(p,dataset.get_calib(frame_idx))  ; a = np.hstack([b.xyz,b.attr[:,0:1]]);a=a.astype(np.float32);a.tofile('/point_gnn/nuscenes_data_check/{:06d}.bin'.format(frame_idx));print(frame_idx);print(len(vertex_coord_list[0]),len(vertex_coord_list[1]))
    '''
    print('-------------------------------------------------')
    for vv,kk,ee in zip(vertex_coord_list,keypoint_indices_list,edges_list):
        print('vertex_list:', np.shape(vv))
        print('keypoint_list:', np.shape(kk))
        print('edge_list:', np.shape(ee))
    print('-------------------------------------------------')
    '''
    if config['input_features'] == 'irgb':
        input_v = cam_rgb_points.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
            cam_rgb_points.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(cam_rgb_points.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([cam_rgb_points.attr[:, [0]],
            np.zeros((cam_rgb_points.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = cam_rgb_points.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
    # From which graph layer the outputs will be gotten
    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    if config['label_method'] == 'yaw':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Car':
        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    if config['label_method'] == 'Pedestrian_and_Cyclist':
        (cls_labels, boxes_3d, valid_boxes, label_map) =\
            dataset.assign_classaware_ped_and_cyc_label_to_points(
            box_label_list, last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz,
        boxes_3d, label_map)
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    #print('Fetch IPython')
    #IPython.embed()
    # np: num_point, nc: num_keypoint or center
    # num_graph_level = 2 usually
    # input_v: (np,1) for reflectance float
    # vertex_coord_list: [(np,3),(nc,3),(nc,3),...]= (num_graph_level+1,),  float
    # keypoint_indices_list: (num_graph_level, nc, 1) int
    # edges_list: [(num_edge_level1,2),(num_edge_level2,2)] = (num_graph_level,): [idx_neighbour,idx_keypoint]
    # cls_labels: (nc,1) int
    # encoded_boxes: (nc,1,7) floats
    # valid_boxes: (nc,1,1) binary
    '''
    print('--------------------------------------------------')
    print('input_v',np.shape(input_v))
    print('vertex_coord_list',np.shape(vertex_coord_list))
    print('keypoint_indices_list',np.shape(keypoint_indices_list))
    print('edges_list',np.shape(edges_list))
    print('cls_labels',np.shape(cls_labels))
    print('encoded_boxes',np.shape(encoded_boxes))
    print('valid_boxes',np.shape(valid_boxes))
    print('input_v',np.shape(input_v[0]))
    print('vertex_coord_list',np.shape(vertex_coord_list[0]))
    print('keypoint_indices_list',np.shape(keypoint_indices_list[0]))
    print('edges_list',np.shape(edges_list[0]))
    print('cls_labels',np.shape(cls_labels[0]))
    print('encoded_boxes',np.shape(encoded_boxes[0]))
    print('valid_boxes',np.shape(valid_boxes[0]))
    print('input_v',input_v[0:2])
    print('vertex_coord_list',vertex_coord_list[0][:2])
    print('keypoint_indices_list',keypoint_indices_list[0][0:2])
    print('edges_list',edges_list[0][0:2])
    print('cls_labels',cls_labels[0:2])
    print('encoded_boxes',encoded_boxes[0:2])
    print('valid_boxes',valid_boxes[0:2])
    print('--------------------------------------------------')
    '''
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)


def return_split_indices(file_path):
    '''
    To return indices in the split file as a list of integer values

    '''
    with open(file_path,'r') as f:
        indices = f.readlines()
        ind_clean = []
        for ind in indices:
            ind_clean.append(int(ind.strip()))
    return ind_clean


Points = namedtuple('Points', ['xyz', 'attr'])
if FLOW_VAL_DATASET_SPLIT_FILE is not None:
    DATA_IDX_SUPERVISED = return_split_indices(DATASET_SPLIT_FILE)
    DATA_IDX_SUPERVISED_VAL = return_split_indices(FLOW_VAL_DATASET_SPLIT_FILE)
    
def fetch_data_flow(frame_idx,pc_idx,is_training=True):
    ''' for supervised flow training data load'''
    if is_training:
        frame_idx = DATA_IDX_SUPERVISED[frame_idx]
        data_path = os.path.join(config['flow_parameters']['flow_supervised_path'],'train')
    else:
        frame_idx = DATA_IDX_SUPERVISED_VAL[frame_idx]
        data_path = os.path.join(config['flow_parameters']['flow_supervised_path'],'test')
    
    data = np.load(os.path.join(data_path,'{:06d}.npz'.format(frame_idx)), allow_pickle=True)
    pc_key = 'pos{}'.format(pc_idx)
    pc = Points(xyz=data[pc_key],attr=np.zeros_like(data[pc_key][:,1:2]))
    gt = data['gt']
    
    
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(pc.xyz, **config['graph_gen_kwargs'])
        
    nump = len(vertex_coord_list[0])
    numc = len(vertex_coord_list[-1])
    if config['input_features'] == 'irgb':
        input_v = pc.attr
    elif config['input_features'] == '0rgb':
        input_v = np.hstack([np.zeros((pc.attr.shape[0], 1)),
            pc.attr[:, 1:]])
    elif config['input_features'] == '0000':
        input_v = np.zeros_like(pc.attr)
    elif config['input_features'] == 'i000':
        input_v = np.hstack([pc.attr[:, [0]],
            np.zeros((pc.attr.shape[0], 3))])
    elif config['input_features'] == 'i':
        input_v = pc.attr[:, [0]]
    elif config['input_features'] == '0':
        input_v = np.zeros((pc.attr.shape[0], 1))
        
    # From which graph layer the outputs will be gotten
    last_layer_graph_level = config['model_kwargs'][
        'layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
    
    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = np.zeros((numc,1))
    encoded_boxes = np.zeros((numc,1,7))
    valid_boxes = np.zeros((numc,1,7))
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    if pc_idx == 1:
        cls_labels = gt
        cls_labels = cls_labels.astype(np.float32) # !!! Gt of the flow (np,3)
        
    
    
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes)



def batch_data(batch_list):
    # batch_list: A list of fetch_data outputs
    # np_i and nc_i are the number of points and keypoints of each batch, respectively
    # N_input_v: (n_batch,np_i,1) A list of attributes with batch_size 
    # N_vertex_coord_list: (n_batch,num_graph_level+1) A list of vertex coordinates in different levels with batch_size
    # N_keypoint_indices_list: (n_batch,num_graph_level,nc_i,1)
    # N_edges_list: (n_batch,num_graph_level)
    # N_cls_labels: (n_batch,nc_i,1)
    # N_encoded_boxes: (n_batch,nc_i,1,7)
    # N_valid_boxes: (n_batch,nc_i,1,1)
    N_input_v, N_vertex_coord_list, N_keypoint_indices_list, N_edges_list,\
    N_cls_labels, N_encoded_boxes, N_valid_boxes = zip(*batch_list)
    
    keypoint_in_each_frame = [len(klist[-1]) for klist in N_keypoint_indices_list] # Num keypoints in each frame of the batch for the last level
    point_in_each_frame = [len(klist[0]) for klist in N_vertex_coord_list] # Num points in each frame of the batch
    
    batch_size = len(batch_list)
    level_num = len(N_vertex_coord_list[0])
    batched_keypoint_indices_list = []
    batched_edges_list = []
    #print('--------------------------------------------------')
    #print('Num of keypoints in batch:',[len(b[-1]) for b in N_keypoint_indices_list])
    #print('--------------------------------------------------')
    '''
    print('--------------------------------------------------')
    print('Shape N_input_v:', np.shape(N_input_v))
    print('Shape N_input_v[0][0]:', np.shape(N_input_v[0][0]))
    print('Shape N_vertex_coord_list: ', np.shape(N_vertex_coord_list))
    print('Shape N_vertex_coord_list[0][0]: ',np.shape(N_vertex_coord_list[0][0]))
    print('Shape N_keypoint_indices_list: ', np.shape(N_keypoint_indices_list))
    print('Shape N_keypoint_indices_list[0]: ', np.shape(N_keypoint_indices_list[0]))
    print('Shape N_edges_list: ', np.shape(N_edges_list))
    print('Shape N_edges_list[0][0]: ',np.shape(N_edges_list[0][0]))
    print('Shape N_cls_labels: ', np.shape(N_cls_labels))
    print('Shape N_cls_labels[0]: ',np.shape(N_cls_labels[0]))
    print('Shape N_encoded_boxes: ', np.shape(N_encoded_boxes))
    print('Shape N_encoded_boxes[0]: ',np.shape(N_encoded_boxes[0]))
    print('Shape N_valid_boxes: ', np.shape(N_valid_boxes))
    print('Shape N_valid_boxes[0][0]: ',np.shape(N_valid_boxes[0][0]))
    print('--------------------------------------------------')
    '''
    for level_idx in range(level_num-1):
        centers = []
        vertices = []
        point_counter = 0
        center_counter = 0
        # Assigns unique point and center indices among batches
        # The points and centers in each batch has different indices from 
        # other batches
        for batch_idx in range(batch_size):
            centers.append(
                N_keypoint_indices_list[batch_idx][level_idx]+point_counter)
            vertices.append(np.hstack(
                [N_edges_list[batch_idx][level_idx][:,[0]]+point_counter,
                 N_edges_list[batch_idx][level_idx][:,[1]]+center_counter]))
            point_counter += N_vertex_coord_list[batch_idx][level_idx].shape[0]
            center_counter += \
                N_keypoint_indices_list[batch_idx][level_idx].shape[0]
        batched_keypoint_indices_list.append(np.vstack(centers))
        batched_edges_list.append(np.vstack(vertices))
    batched_vertex_coord_list = []
    # Get points of the same level from different batches together
    for level_idx in range(level_num):
        points = []
        counter = 0
        for batch_idx in range(batch_size):
            points.append(N_vertex_coord_list[batch_idx][level_idx])
        batched_vertex_coord_list.append(np.vstack(points))
    batched_input_v = np.vstack(N_input_v)
    batched_cls_labels = np.vstack(N_cls_labels)
    batched_encoded_boxes = np.vstack(N_encoded_boxes)
    batched_valid_boxes = np.vstack(N_valid_boxes)
    #print('batch IPython')
    #IPython.embed()
    '''
    print('--------------------------------------------------')
    print('Shape batched_input_v:', np.shape(batched_input_v))
    print('Shape batched_input_v[0]:', np.shape(batched_input_v[0]))
    print('Shape batched_vertex_coord_list: ', np.shape(batched_vertex_coord_list))
    print('Shape batched_vertex_coord_list[0]: ',np.shape(batched_vertex_coord_list[0]))
    print('Shape batched_keypoint_indices_list: ', np.shape(batched_keypoint_indices_list))
    print('Shape batched_keypoint_indices_list[0]: ', np.shape(batched_keypoint_indices_list[0]))
    print('Shape batched_edges_list: ', np.shape(batched_edges_list))
    print('Shape batched_edges_list[0]: ',np.shape(batched_edges_list[0]))
    print('Shape batched_cls_labels: ', np.shape(batched_cls_labels))
    print('Shape batched_cls_labels[0]: ',np.shape(batched_cls_labels[0]))
    print('Shape batched_encoded_boxes: ', np.shape(batched_encoded_boxes))
    print('Shape batched_encoded_boxes[0]: ',np.shape(batched_encoded_boxes[0]))
    print('Shape batched_valid_boxes: ', np.shape(batched_valid_boxes))
    print('Shape batched_valid_boxes[0]: ',np.shape(batched_valid_boxes[0]))
    print('--------------------------------------------------')
    '''
    # ne1_i: number of edges in the 1st graph level of batch i, ne2_i: the same for the 2nd graph level
    # batched_input_v: (n_batch*np_i,1)
    # batched_vertex_coord_list: (num_graph_level+1,): [(n_batch*np_i,3),(n_batch*nc_i,3),(n_batch*nc_i,3)]
    # batched_keypoint_indices_list: (num_graph_level,n_batch*nc_i,1)
    # batched_edges_list: (num_graph_level,): [(n_batch*ne1_i,2),(n_batch,ne2_i,2)]
    # batched_cls_labels: (n_batch*nc_i,1)
    # batched_encoded_boxes: (n_batch*nc_i,1,7)
    # batched_valid_boxes: (n_batch*nc_i,1,1)
    return (batched_input_v, batched_vertex_coord_list,
        batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
        batched_encoded_boxes, batched_valid_boxes,keypoint_in_each_frame,point_in_each_frame)

# optimizers ================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
BN_INIT_DECAY = 0.5
BATCH_SIZE = train_config.get('batch_size', 1)
BN_DECAY_DECAY_STEP = 200000.
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.99
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

# model =======================================================================
if 'COPY_PER_GPU' in train_config:
    COPY_PER_GPU = train_config['COPY_PER_GPU']
else:
    COPY_PER_GPU = 1
NUM_GPU = train_config['NUM_GPU']
input_tensor_sets = []
for gi in range(NUM_GPU):
    with tf.device('/gpu:%d'%gi):
        for cp_idx in range(COPY_PER_GPU):
            # t_initial_vertex_features: float32 [None,1]
            # t_vertex_coord_list: float32 [[None,3],[None,3],[None,3]]
            # t_edges_list: int32 [[None,None],[None,None]]
            # t_keypoint_indices_list: int32 [[None,1],[None,1]]
            # t_class_labels: int32 [None,1]
            # t_encoded_gt_boxes: float32 [None,1,7]
            # t_valid_gt_boxes: float32 [None,1,1]
            if config['input_features'] == 'irgb':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'rgb':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 3])
            elif config['input_features'] == '0000':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'i000':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 4])
            elif config['input_features'] == 'i':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1])
            elif config['input_features'] == '0':
                t_initial_vertex_features = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1])

            t_vertex_coord_list = [
                tf.placeholder(dtype=tf.float32, shape=[None, 3])]
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
                t_vertex_coord_list.append(
                    tf.placeholder(dtype=tf.float32, shape=[None, 3]))

            t_edges_list = []
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
                t_edges_list.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, None]))

            t_keypoint_indices_list = []
            for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
                t_keypoint_indices_list.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, 1]))
            
            
            t_n_keypoints = tf.placeholder(dtype=tf.int32, shape=[None],name='n_keypoints_pc1')
            
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                t_class_labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
                
                t_encoded_gt_boxes = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1, BOX_ENCODING_LEN])
                t_valid_gt_boxes = tf.placeholder(
                    dtype=tf.float32, shape=[None, 1, 1])
            else:
                t_class_labels = None
                t_encoded_gt_boxes = None
                t_valid_gt_boxes = None
            
            
            
            if TRAIN_OPT == 'flow' and config['flow_parameters']['supervised']:
                t_flow_labels = tf.placeholder(dtype=tf.float32, shape=[None, 3])
            else:
                t_flow_labels = None
            
            ## Placeholders for the 2nd point cloud
            ## if flow train is not selected, returns None
            if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                pc2_t_initial_vertex_features, pc2_t_vertex_coord_list, \
                    pc2_t_keypoint_indices_list, pc2_t_edges_list, \
                    pc2_t_class_labels, pc2_t_encoded_gt_boxes, \
                    pc2_t_valid_gt_boxes,pc2_t_n_keypoints, \
                    pc2_t_n_points = get_flow_placeholders(config,BOX_ENCODING_LEN)
                t_n_points = tf.placeholder(dtype=tf.int32, shape=[None],name='n_points_pc1')
            else:
                pc2_t_initial_vertex_features, pc2_t_vertex_coord_list, \
                    pc2_t_keypoint_indices_list, pc2_t_edges_list, \
                    pc2_t_class_labels, pc2_t_encoded_gt_boxes, \
                    pc2_t_valid_gt_boxes,pc2_t_n_keypoints, \
                    pc2_t_n_points = [None] * 9
                t_n_points = None
            ################
            
            t_is_training = tf.placeholder(dtype=tf.bool, shape=[])
            bn_decay = get_bn_decay(global_step)
            model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
                box_encoding_len=BOX_ENCODING_LEN, mode='train', 
                batch_size=train_config.get('batch_size', 1)//(COPY_PER_GPU*NUM_GPU),
                **config['model_kwargs'],global_step=global_step,bn_decay=bn_decay,
                flow_parameters=config['flow_parameters'],repeated_batch_flow=REPEATED_BATCH_FLOW)
            
            
            t_logits, t_pred_box,flow_tensors = model.predict(
                t_initial_vertex_features, t_vertex_coord_list,
                t_keypoint_indices_list, t_edges_list, t_is_training,
                pc2_t_initial_vertex_features, pc2_t_vertex_coord_list,
                pc2_t_keypoint_indices_list, pc2_t_edges_list,t_n_keypoints,
                pc2_t_n_keypoints, t_n_points, pc2_t_n_points,t_flow_labels)            
            
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                t_probs = model.postprocess(t_logits)
                t_predictions = tf.argmax(t_probs, axis=-1, output_type=tf.int32)
                t_loss_dict = model.loss(t_logits, t_class_labels, t_pred_box,
                    t_encoded_gt_boxes, t_valid_gt_boxes, **config['loss'])
                t_cls_loss = t_loss_dict['cls_loss']
                t_loc_loss = t_loss_dict['loc_loss']
                t_reg_loss = t_loss_dict['reg_loss']
            if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                t_flow_loss = flow_tensors['flow_loss']
            # num_endpoint is the number of output vertices.
            # num_valid_endpoint is the number of output vertices that have a valid
            # bounding box. Those numbers are useful for weighting during batching.
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                t_num_endpoint = t_loss_dict['num_endpoint']
                t_num_valid_endpoint = t_loss_dict['num_valid_endpoint']
                t_classwise_loc_loss = t_loss_dict['classwise_loc_loss']
                t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss
            input_tensor_sets.append(
                {'t_initial_vertex_features': t_initial_vertex_features,
                 't_vertex_coord_list': t_vertex_coord_list,
                 't_edges_list':t_edges_list,
                 't_keypoint_indices_list': t_keypoint_indices_list,
                 't_n_keypoints':t_n_keypoints,
                 't_n_points':t_n_points,
                 't_is_training': t_is_training,
                 })

            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                input_tensor_sets[-1]['t_logits']=t_logits
                input_tensor_sets[-1]['t_pred_box']=t_pred_box
                input_tensor_sets[-1]['t_probs']=t_probs
                input_tensor_sets[-1]['t_predictions']=t_predictions
                input_tensor_sets[-1]['t_cls_loss']=t_cls_loss
                input_tensor_sets[-1]['t_loc_loss']=t_loc_loss
                input_tensor_sets[-1]['t_reg_loss']=t_reg_loss
                input_tensor_sets[-1]['t_num_endpoint']=t_num_endpoint
                input_tensor_sets[-1]['t_num_valid_endpoint']=t_num_valid_endpoint
                input_tensor_sets[-1]['t_classwise_loc_loss']=t_classwise_loc_loss
                input_tensor_sets[-1]['t_total_loss']=t_total_loss
                input_tensor_sets[-1]['t_class_labels']=t_class_labels
                input_tensor_sets[-1]['t_encoded_gt_boxes']=t_encoded_gt_boxes
                input_tensor_sets[-1]['t_valid_gt_boxes']=t_valid_gt_boxes
            if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                input_tensor_sets[-1]['pc2_t_initial_vertex_features']=pc2_t_initial_vertex_features
                input_tensor_sets[-1]['pc2_t_vertex_coord_list']=pc2_t_vertex_coord_list
                input_tensor_sets[-1]['pc2_t_keypoint_indices_list']=pc2_t_keypoint_indices_list
                input_tensor_sets[-1]['pc2_t_edges_list']=pc2_t_edges_list
                input_tensor_sets[-1]['pc2_t_class_labels']=pc2_t_class_labels
                input_tensor_sets[-1]['pc2_t_encoded_gt_boxes']=pc2_t_encoded_gt_boxes
                input_tensor_sets[-1]['pc2_t_valid_gt_boxes']=pc2_t_valid_gt_boxes
                input_tensor_sets[-1]['pc2_t_n_keypoints']=pc2_t_n_keypoints
                input_tensor_sets[-1]['pc2_t_n_points']=pc2_t_n_points
                input_tensor_sets[-1]['t_flow_loss']=t_flow_loss
            if TRAIN_OPT == 'flow' and config['flow_parameters']['supervised']:
                input_tensor_sets[-1]['t_flow_labels']=t_flow_labels

if 'unify_copies' in train_config:
    if train_config['unify_copies']:
        # re-weight loss for the number of end points
        print('Set to unify copies in different GPU as if its a single copy')
        if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
            total_num_endpoints = tf.reduce_sum([t['t_num_endpoint']
                for t in input_tensor_sets])
            total_num_valid_endpoints = tf.reduce_sum([t['t_num_valid_endpoint']
                for t in input_tensor_sets])
        for ti in range(len(input_tensor_sets)):
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                weight = tf.div_no_nan(
                    tf.cast(len(input_tensor_sets)*input_tensor_sets[ti][
                        't_num_endpoint'], tf.float32),
                    tf.cast(total_num_endpoints, tf.float32))
                weight = tf.cast(weight, tf.float32)
                valid_weight = tf.div_no_nan(
                    tf.cast(len(input_tensor_sets)*input_tensor_sets[ti][
                        't_num_valid_endpoint'], tf.float32),
                    tf.cast(total_num_valid_endpoints, tf.float32))
                valid_weight = tf.cast(valid_weight, tf.float32)
                input_tensor_sets[ti]['t_cls_loss'] *= weight
                input_tensor_sets[ti]['t_loc_loss'] *= valid_weight
                det_losses = input_tensor_sets[ti]['t_cls_loss']\
                                +input_tensor_sets[ti]['t_loc_loss']\
                                +input_tensor_sets[ti]['t_reg_loss']
            if TRAIN_OPT == '3Ddet':
                input_tensor_sets[ti]['t_total_loss'] = det_losses 
            elif TRAIN_OPT == 'flow':
                input_tensor_sets[ti]['t_total_loss'] = \
                    input_tensor_sets[ti]['t_flow_loss'] * FLOW_LOSS_WEIGHT
            else:
                input_tensor_sets[ti]['t_total_loss'] = det_losses  \
                    +input_tensor_sets[ti]['t_flow_loss'] * FLOW_LOSS_WEIGHT

if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
    t_cls_loss_cross_gpu = tf.reduce_mean([t['t_cls_loss']
        for t in input_tensor_sets])
    t_loc_loss_cross_gpu = tf.reduce_mean([t['t_loc_loss']
        for t in input_tensor_sets])
    t_reg_loss_cross_gpu = tf.reduce_mean([t['t_reg_loss']
        for t in input_tensor_sets])
    
    t_class_labels = input_tensor_sets[0]['t_class_labels']
    t_predictions = input_tensor_sets[0]['t_predictions']
    t_probs = input_tensor_sets[0]['t_probs']
  
if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':  
    t_flow_loss_cross_gpu = tf.reduce_mean([t['t_flow_loss']
        for t in input_tensor_sets])
    
t_total_loss_cross_gpu = tf.reduce_mean([t['t_total_loss']
        for t in input_tensor_sets])


if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
    t_classwise_loc_loss_update_ops = {}
    for class_idx in range(NUM_CLASSES):
        for bi in range(BOX_ENCODING_LEN):
            classwise_loc_loss_ind =tf.reduce_sum(
                [input_tensor_sets[gi]['t_classwise_loc_loss'][class_idx][bi]
                    for gi in range(len(input_tensor_sets))])
            t_mean_loss, t_mean_loss_op = tf.metrics.mean(
                classwise_loc_loss_ind,
                name=('loc_loss_cls_%d_box_%d'%(class_idx, bi)))
            t_classwise_loc_loss_update_ops[
                ('loc_loss_cls_%d_box_%d'%(class_idx, bi))] = t_mean_loss_op
        classwise_loc_loss =tf.reduce_sum(
            [input_tensor_sets[gi]['t_classwise_loc_loss'][class_idx]
                for gi in range(len(input_tensor_sets))])
        t_mean_loss, t_mean_loss_op = tf.metrics.mean(
            classwise_loc_loss,
            name=('loc_loss_cls_%d'%class_idx))
        t_classwise_loc_loss_update_ops[
            ('loc_loss_cls_%d'%class_idx)] = t_mean_loss_op


# metrics
if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
    t_recall_update_ops = {}
    for class_idx in range(NUM_CLASSES):
        t_recall, t_recall_update_op = tf.metrics.recall(
            tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
            tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
            name=('recall_%d'%class_idx))
        t_recall_update_ops[('recall_%d'%class_idx)] = t_recall_update_op
    
    t_precision_update_ops = {}
    for class_idx in range(NUM_CLASSES):
        t_precision, t_precision_update_op = tf.metrics.precision(
            tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
            tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
            name=('precision_%d'%class_idx))
        t_precision_update_ops[('precision_%d'%class_idx)] = t_precision_update_op
    
    t_mAP_update_ops = {}
    for class_idx in range(NUM_CLASSES):
        t_mAP, t_mAP_update_op = tf.metrics.auc(
            tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
            t_probs[:, class_idx],
            num_thresholds=200,
            curve='PR',
            name=('mAP_%d'%class_idx),
            summation_method='careful_interpolation')
        t_mAP_update_ops[('mAP_%d'%class_idx)] = t_mAP_update_op
    
    t_mean_cls_loss, t_mean_cls_loss_op = tf.metrics.mean(
        t_cls_loss_cross_gpu,
        name='mean_cls_loss')
    t_mean_loc_loss, t_mean_loc_loss_op = tf.metrics.mean(
        t_loc_loss_cross_gpu,
        name='mean_loc_loss')
    t_mean_reg_loss, t_mean_reg_loss_op = tf.metrics.mean(
        t_reg_loss_cross_gpu,
        name='mean_reg_loss')

if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
    t_mean_flow_loss, t_mean_flow_loss_op = tf.metrics.mean(
        t_flow_loss_cross_gpu, name='mean_flow_loss')
    
t_mean_total_loss, t_mean_total_loss_op = tf.metrics.mean(
    t_total_loss_cross_gpu, name='mean_total_loss')


metrics_update_ops = dict()
if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
    metrics_update_ops['cls_loss']= t_mean_cls_loss_op
    metrics_update_ops['loc_loss']= t_mean_loc_loss_op
    metrics_update_ops['reg_loss']= t_mean_reg_loss_op
    metrics_update_ops.update(t_recall_update_ops)
    metrics_update_ops.update(t_precision_update_ops)
    metrics_update_ops.update(t_mAP_update_ops)
    metrics_update_ops.update(t_classwise_loc_loss_update_ops)

if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
    metrics_update_ops['flow_loss']= t_mean_flow_loss_op

metrics_update_ops['total_loss']= t_mean_total_loss_op

# optimizers ================================================================
print("LEARNING RATE: ", train_config['initial_lr'])
t_learning_rate = tf.train.exponential_decay(train_config['initial_lr'],
    global_step, train_config['decay_step'], train_config['decay_factor'],
    staircase=train_config.get('is_staircase', True))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer_dict = {
    'sgd': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'rmsprop':  tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
}
optimizer_kwargs_dict = {
    'sgd': {},
    'momentum': {'momentum': 0.9},
    'rmsprop':  {'momentum': 0.9, 'decay': 0.9, 'epsilon': 1.0},
    'adam': {}
}
optimizer_class = optimizer_dict[train_config['optimizer']]
optimizer_kwargs = optimizer_kwargs_dict[train_config['optimizer']]
if 'optimizer_kwargs' in train_config:
    optimizer_kwargs.update(train_config['optimizer_kwargs'])
optimizer = optimizer_class(t_learning_rate, **optimizer_kwargs)
grads_cross_gpu = []
with tf.control_dependencies(update_ops):
    for gi in range(NUM_GPU):
        with tf.device('/gpu:%d'%gi):
            grads = optimizer.compute_gradients(
                input_tensor_sets[gi]['t_total_loss'])
            grads_cross_gpu.append(grads)
grads_cross_gpu = average_gradients(grads_cross_gpu)
train_op = optimizer.apply_gradients(grads_cross_gpu, global_step=global_step)
fetches = {
    'train_op': train_op,
    'step': global_step,
    'learning_rate': t_learning_rate,
}
fetches.update(metrics_update_ops)

if TRAIN_OPT == 'flow':
    fetches['flow_tensors']=flow_tensors
    keys_to_remove = []
    for k in fetches['flow_tensors'].keys():
        if fetches['flow_tensors'][k] is None:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del fetches['flow_tensors'][k]
    #if config['flow_parameters']['supervised']:
    #    del fetches['flow_tensors']['flow_end_points_loss']




# TODO: Update data provider according to the task given with TRAIN_OPT
class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, dataset should be
    reloaded.
    """
    def __init__(self, fetch_data, batch_data, load_dataset_to_mem=True,
        load_dataset_every_N_time=1, capacity=1, num_workers=1, preload_list=[],
        async_load_rate=1.0, result_pool_limit=10000, flow_supervised=False,
        fetch_data_flow=None,is_training=True,dataset=None,train_opt='both'):
        self._fetch_data = fetch_data
        self._batch_data = batch_data
        self._buffer = {}
        self._results = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
        self._worker_pool = Pool(processes=num_workers)
        self._preload_list = preload_list
        self._async_load_rate = async_load_rate
        self._result_pool_limit = result_pool_limit
        self._dataset = dataset
        self._train_opt = train_opt
        self._flow_supervised = flow_supervised
        if len(self._preload_list) > 0:
            self.preload(self._preload_list)
        if self._flow_supervised:
            self._fetch_data_flow = fetch_data_flow
        self._is_training = is_training
        
    def flush_data(self):
        del self._results
        self._results = {}
    def preload(self, frame_idx_list):
        """async load dataset into memory."""
        if not self._flow_supervised:
            for frame_idx in frame_idx_list:
                result = self._worker_pool.apply_async(
                    self._fetch_data, (frame_idx,self._dataset,))
                self._results[frame_idx] = result
        else:
            pass

    def async_load(self, frame_idx):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            if self._is_training:
                del self._results[frame_idx]
            else:
                try:
                    del self._results[frame_idx-20]
                except:
                    pass
        else:
            #print('{} not in'.format(frame_idx))
            data = self._fetch_data(frame_idx,self._dataset)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                if self._is_training:
                    result = self._worker_pool.apply_async(
                        self._fetch_data, (frame_idx,self._dataset,))
                    self._results[frame_idx] = result
        return data
    
    def async_load_flow(self, frame_idx,pc_id):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            del self._results[frame_idx]
        else:
            #print('{} not in'.format(frame_idx))
            data = self._fetch_data_flow(frame_idx,pc_id,self._is_training)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                if self._is_training:
                    result = self._worker_pool.apply_async(
                        self._fetch_data_flow, (frame_idx,pc_id,self._is_training))
                    self._results[frame_idx] = result
        return data

    def provide(self, frame_idx,pc_id=None):
        if not self._flow_supervised:
            if self._load_dataset_to_mem:
                if self._load_every_N_time >= 1:
                    extend_frame_idx = frame_idx+np.random.choice(
                        self._capacity)*NUM_TEST_SAMPLE
                    if extend_frame_idx not in self._buffer:
                        data = self.async_load(frame_idx)
                        self._buffer[extend_frame_idx] = (data, 0)
                    data, ctr = self._buffer[extend_frame_idx]
                    if ctr == self._load_every_N_time:
                        data = self.async_load(frame_idx)
                        self._buffer[extend_frame_idx] = (data, 0)
                    data, ctr = self._buffer[extend_frame_idx]
                    self._buffer[extend_frame_idx] = (data, ctr+1)
                    return data
                else:
                    # do not buffer
                    return self.async_load(frame_idx)
            else:
                return self._fetch_data(frame_idx,self._dataset)
        else:
            if self._load_dataset_to_mem:
                return self.async_load_flow(frame_idx,pc_id)
            else:
                return self._fetch_data_flow(frame_idx,pc_id,self._is_training)
        
          

    def provide_batch(self, frame_idx_list):
        st=time.time()
        batch_list = []
        batch_list_pre = [] # One previous frames of the frrames in frame_idx_list
        if not self._flow_supervised:
            for frame_idx in frame_idx_list:
                batch_list.append(self.provide(frame_idx))
                if self._train_opt == 'flow' or self._train_opt == 'both':
                    batch_list_pre.append(self.provide(frame_idx-1))
        else:
            for frame_idx in frame_idx_list:
                batch_list.append(self.provide(frame_idx,1))
                batch_list_pre.append(self.provide(frame_idx,2))
        print('one batch time:',time.time()-st)
        if self._train_opt == 'flow' or self._train_opt == 'both':
            return self._batch_data(batch_list),self._batch_data(batch_list_pre)
        else:
            return self._batch_data(batch_list),None


#if train_config['load_dataset_to_mem']:
#    preload_list = list(range(NUM_TEST_SAMPLE))
#else:
#    preload_list = []

if 'pool_limit' in train_config:
    pool_limit = train_config['pool_limit']
else:
    pool_limit = 10000
    
preload_list = list(range(min(NUM_TEST_SAMPLE,pool_limit)))

data_provider = DataProvider(fetch_data, batch_data,
    load_dataset_to_mem=train_config['load_dataset_to_mem'],
    load_dataset_every_N_time=train_config['load_dataset_every_N_time'],
    capacity=train_config['capacity'],
    num_workers=train_config['num_load_dataset_workers'],
    preload_list=preload_list,
    flow_supervised=config['flow_parameters']['supervised'],
    fetch_data_flow = fetch_data_flow,dataset=dataset,
    result_pool_limit=pool_limit,train_opt=TRAIN_OPT)

## Prepare validation data here
if FLOW_VAL_DATASET_SPLIT_FILE is not None:
    '''
    if train_config['load_dataset_to_mem']:
        preload_list = list(range(len(DATA_IDX_SUPERVISED_VAL)))
    else:
        preload_list = []
    '''
    if 'pool_limit' in train_config:
        pool_limit = train_config['pool_limit']
    else:
        pool_limit = 10000
    
    preload_list = list(range(min(len(DATA_IDX_SUPERVISED_VAL),pool_limit)))
    
    data_provider_eval = DataProvider(fetch_data, batch_data,
        load_dataset_to_mem=train_config['load_dataset_to_mem'],
        load_dataset_every_N_time=train_config['load_dataset_every_N_time'],
        capacity=train_config['capacity'],
        num_workers=train_config['num_load_dataset_workers'],
        preload_list=preload_list,
        flow_supervised=config['flow_parameters']['supervised'],
        fetch_data_flow = fetch_data_flow, is_training=False,
        dataset=dataset_val,result_pool_limit=pool_limit,
        train_opt=TRAIN_OPT)
    # In validation, we don't fetch train_op from the tf.Session
    fetches_eval = fetches.copy()
    fetches_eval.pop('train_op',None)
# Training session ==========================================================
batch_size = train_config.get('batch_size', 1)
print('batch size=' + str(batch_size))
saver = tf.train.Saver(max_to_keep=15,keep_checkpoint_every_n_hours=2)
graph = tf.get_default_graph()
if train_config['gpu_memusage'] < 0:
    gpu_options = tf.GPUOptions(allow_growth=True)
else:
    if train_config['gpu_memusage'] < -10:
        gpu_options = tf.GPUOptions()
    else:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=train_config['gpu_memusage'])
batch_ctr = 0
batch_gradient_list = []

def eval_one_step():
    start_time = time.time()
    val_num_test_sample = len(DATA_IDX_SUPERVISED_VAL)
    frame_idx_list = np.arange(val_num_test_sample)
    mean_epe = []
    mean_acc1 = []
    mean_acc2 = []
    mean_error = []
    mean_eval = []
    for batch_idx in range(0, val_num_test_sample-batch_size+1, batch_size):
        device_batch_size = batch_size//(COPY_PER_GPU*NUM_GPU)
        total_feed_dict = {}
        
        ##### Next batch indices for preloading
        for gi in range(COPY_PER_GPU*NUM_GPU):
            bb = (batch_idx+10)%val_num_test_sample
            nb = frame_idx_list[bb+gi*device_batch_size:bb+(gi+1)*device_batch_size]
            nbb= []
            for _nb in nb:
                nbb.append(_nb)
                if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                    nbb.append(_nb-1)
            #print('preload ids {}'.format(nbb))
            data_provider_eval.preload(nbb)
            
        
        ### Start evaluation
        for gi in range(COPY_PER_GPU*NUM_GPU):
            batch_frame_idx_list = frame_idx_list[
                batch_idx+\
                gi*device_batch_size:batch_idx+(gi+1)*device_batch_size]
            
            batch_data_pc1, batch_data_pc2 = data_provider_eval.provide_batch(batch_frame_idx_list)
            # Get data of main point cloud (pc1)
            input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                cls_labels, encoded_boxes, valid_boxes, keypoint_batch_pc1, \
                point_batch_pc1 = batch_data_pc1
            # TODO: Don't get pc2 for 3Ddet
            # Get data of point clouds of previous frames (pc2)
            if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                pc2_input_v, pc2_vertex_coord_list, pc2_keypoint_indices_list,\
                    pc2_edges_list, pc2_cls_labels, pc2_encoded_boxes,\
                    pc2_valid_boxes, keypoint_batch_pc2, point_batch_pc2 = batch_data_pc2
            
            ### Feed dict for PC1
            t_initial_vertex_features = \
                input_tensor_sets[gi]['t_initial_vertex_features']
            t_is_training = input_tensor_sets[gi]['t_is_training']
            t_edges_list = input_tensor_sets[gi]['t_edges_list']
            t_keypoint_indices_list = \
                input_tensor_sets[gi]['t_keypoint_indices_list']
            t_vertex_coord_list = \
                input_tensor_sets[gi]['t_vertex_coord_list']
            t_n_keypoints = input_tensor_sets[gi]['t_n_keypoints']
            t_n_points = input_tensor_sets[gi]['t_n_points']
            
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                t_class_labels = input_tensor_sets[gi]['t_class_labels']
                t_encoded_gt_boxes = input_tensor_sets[gi]['t_encoded_gt_boxes']
                t_valid_gt_boxes = input_tensor_sets[gi]['t_valid_gt_boxes']
            
            
            feed_dict = {
                t_initial_vertex_features: input_v,
                t_is_training: False,
                t_n_keypoints: keypoint_batch_pc1
            }
            feed_dict.update(dict(zip(t_edges_list, edges_list)))
            feed_dict.update(
                dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
            feed_dict.update(
                dict(zip(t_vertex_coord_list, vertex_coord_list)))
            
            if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                feed_dict[t_class_labels] = cls_labels
                feed_dict[t_encoded_gt_boxes] = encoded_boxes
                feed_dict[t_valid_gt_boxes] = valid_boxes
                
            ### Feed dict for PC2
            if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                pc2_t_initial_vertex_features = \
                    input_tensor_sets[gi]['pc2_t_initial_vertex_features']
                pc2_t_class_labels = input_tensor_sets[gi]['pc2_t_class_labels']
                pc2_t_encoded_gt_boxes = input_tensor_sets[gi]['pc2_t_encoded_gt_boxes']
                pc2_t_valid_gt_boxes = input_tensor_sets[gi]['pc2_t_valid_gt_boxes']
                pc2_t_edges_list = input_tensor_sets[gi]['pc2_t_edges_list']
                pc2_t_keypoint_indices_list = \
                    input_tensor_sets[gi]['pc2_t_keypoint_indices_list']
                pc2_t_vertex_coord_list = \
                    input_tensor_sets[gi]['pc2_t_vertex_coord_list']
                pc2_t_n_keypoints = input_tensor_sets[gi]['pc2_t_n_keypoints']
                pc2_t_n_points = input_tensor_sets[gi]['pc2_t_n_points']
                feed_dict[pc2_t_initial_vertex_features] = pc2_input_v
                feed_dict[pc2_t_n_keypoints] = keypoint_batch_pc2
                feed_dict[pc2_t_n_points] = point_batch_pc2
                feed_dict[t_n_points] =  point_batch_pc1
                feed_dict.update(dict(zip(pc2_t_edges_list, pc2_edges_list)))
                feed_dict.update(
                    dict(zip(pc2_t_keypoint_indices_list, pc2_keypoint_indices_list)))
                feed_dict.update(
                    dict(zip(pc2_t_vertex_coord_list, pc2_vertex_coord_list)))
            if TRAIN_OPT == 'flow' and config['flow_parameters']['supervised']:
                input_tensor_sets[gi]['t_flow_labels']
                feed_dict[t_flow_labels] = cls_labels
                
            total_feed_dict.update(feed_dict)

        results = sess.run(fetches_eval, feed_dict=total_feed_dict)
        
        if TRAIN_OPT == 'flow':
            print('Eval flow:%f' % (results['flow_loss']))
            mean_eval.append(results['flow_loss'])
        print('STEP: %d / %d, time cost: %f'
            % (batch_idx,val_num_test_sample,time.time()-start_time))
        
        if config['flow_parameters']['supervised']:
            epe, acc1, acc2, error, gt_label = scene_flow_EPE_np(results['flow_tensors']['pred'],
                                                             results['flow_tensors']['labels'],
                                        np.ones(results['flow_tensors']['pred'].shape, dtype=np.int32)[:,:,0])
            mean_epe.append(epe)
            mean_acc1.append(acc1)
            mean_acc2.append(acc2)
            mean_error.append(error)

    if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
        print('Eval flow:%f' % (results['flow_loss']))
    print('STEP: %d, time cost: %f'
            % (results['step'],time.time()-start_time))
    for key in metrics_update_ops:
        write_summary_scale(key, results[key], results['step'],
                os.path.join(train_config['train_dir'],'test'))
    if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
        write_summary_scale('mean_flow_eval',np.mean(mean_eval),results['step'],
                os.path.join(train_config['train_dir'],'test'))
    
    
    if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
        for class_idx in range(NUM_CLASSES):
            print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                % (class_idx,
                results['recall_%d'%class_idx],
                results['precision_%d'%class_idx],
                results['mAP_%d'%class_idx],
                results['loc_loss_cls_%d'%class_idx]))
            print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
            %(
            results['loc_loss_cls_%d_box_%d'%(class_idx, 0)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 1)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 2)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 3)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 4)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 5)],
            results['loc_loss_cls_%d_box_%d'%(class_idx, 6)]),
            )

    
    if config['flow_parameters']['supervised']:    
        write_summary_scale('epe',np.mean(mean_epe),results['step'],
                    os.path.join(train_config['train_dir'],'test'))
        write_summary_scale('acc1',np.mean(mean_acc1),results['step'],
                    os.path.join(train_config['train_dir'],'test'))
        write_summary_scale('acc2',np.mean(mean_acc2),results['step'],
                    os.path.join(train_config['train_dir'],'test'))
        write_summary_scale('error',np.mean(mean_error),results['step'],
                    os.path.join(train_config['train_dir'],'test'))

    data_provider_eval.flush_data()
    return results['total_loss']
    
def get_latest_checkpoint(ckpt_path):
    if type(ckpt_path) != str:
        return None
    path = os.path.join(ckpt_path,"checkpoint")
    if os.path.isfile(path):
        with open(path,'r') as f:
            ckpts = f.readlines()
            ckpts.reverse() # last record points to the latest checkpoint
            for c in ckpts:
                c = c.strip() # one line
                d = c.split(': ')[1][1:-1] # path to the model ckpt
                model_name = os.path.basename(d) # model_name
                if '-' in model_name:
                    return d
    else:
        return None
def restore_weights(sess,config):
    global detection_head, backbone, scene_flow_head
    print('Restore Weights')
    consider_latest = config.get('alternating_train',False) # If true, the latest ckpt in the path will be taken
    
    backbone_restore_path = config.get("backbone_restore",None)
    curr_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    if consider_latest:
        backbone_restore_path = get_latest_checkpoint(backbone_restore_path)
    if backbone_restore_path is not None:
        print('Backbone restore start')
        filtered_backbone_list = [v for v in curr_variables \
                                  if v.name.split(':')[0] in backbone]
        saver2 = tf.train.Saver(max_to_keep=50,var_list=filtered_backbone_list)
        saver2.restore(sess, backbone_restore_path)
        print('Backbone restore done!')
    
    detection_restore_path = config.get("detection_head_restore",None)
    if consider_latest:
        detection_restore_path = get_latest_checkpoint(detection_restore_path)
        detection_head += ['Variable']
    if detection_restore_path is not None and (config['train'] == '3Ddet' or config['train'] == 'both'):
        print('Detection head restore start')
        filtered_detection_list = [v for v in curr_variables \
                                  if v.name.split(':')[0] in detection_head]
        saver2 = tf.train.Saver(max_to_keep=50,var_list=filtered_detection_list)
        saver2.restore(sess, detection_restore_path)
        print('Detection head restore done!')
    
    scene_flow_restore_path = config.get("scene_flow_head_restore",None)
    if consider_latest:
        scene_flow_restore_path = get_latest_checkpoint(scene_flow_restore_path)
        scene_flow_head += ['Variable']
    if scene_flow_restore_path is not None and (config['train'] == 'flow' or config['train'] == 'both'):
        print('Scene flow head restore start')
        filtered_scene_flow_list = [v for v in curr_variables \
                                  if v.name.split(':')[0] in scene_flow_head]
        saver2 = tf.train.Saver(max_to_keep=50,var_list=filtered_scene_flow_list)
        saver2.restore(sess, scene_flow_restore_path)
        print('Scene flow head restore done!')
    
    print('Restoring weights done!')
    
    

def restore_flow(sess,path):
    print('restore')
    vars_in_checkpoint = tf.train.list_variables(path)
    var_list = [v[0] for v in vars_in_checkpoint]
    curr_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    if 'filter_layers' not in config['flow_parameters']:
        config['flow_parameters']['filter_layers']=True
        
    if config['flow_parameters']['filter_layers']:
        filtered_curr_vars = [v for v in curr_variables \
                         if v.name.split(':')[0] in var_list and \
                            'flownet3d/sa1/layer2/conv0' not in v.name.split(':')[0] and \
                            'flownet3d/layer4/conv0' not in v.name.split(':')[0] and \
                            'flownet3d/up_sa_layer2/conv0/weights:0' not in v.name.split(':')[0] and \
                            'flownet3d/up_sa_layer1/post-conv0' not in v.name.split(':')[0] and \
                            'flownet3d/fa_layer4/conv_0' not in v.name.split(':')[0] ]
    else:
        filtered_curr_vars = [v for v in curr_variables \
                         if v.name.split(':')[0] in var_list ]
    saver2 = tf.train.Saver(max_to_keep=50,var_list=filtered_curr_vars)
    saver2.restore(sess, path)



min_eval_loss = read_min_loss(os.path.join(train_config['train_dir'],'best','min_eval_loss'))
with tf.Session(graph=graph,
    config=tf.ConfigProto(
    allow_soft_placement=True, gpu_options=gpu_options,)) as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    # TODO: Restore from the given path for the flow and Point-GNN
    if FLOW_RESTORE is not None:
        restore_flow(sess,FLOW_RESTORE)
    
    restore_weights(sess,config['flow_parameters'])
    # Checks the previous states in the train_dir and restores if there is something
    states = tf.train.get_checkpoint_state(train_config['train_dir'])
    if states is not None and not config['flow_parameters'].get('alternating_train',False):
        print('Restore from checkpoint %s' % states.model_checkpoint_path)
        saver.restore(sess, states.model_checkpoint_path)
        if config['flow_parameters']['train'] != 'flow':
            saver.recover_last_checkpoints(states.all_model_checkpoint_paths)
    

    previous_step = sess.run(global_step)
    if config['flow_parameters'].get('alternating_train',False):
        train_config['max_epoch'] = int((previous_step*batch_size)//NUM_TEST_SAMPLE + train_config['alternating_epoch'])
    local_variables_initializer = tf.variables_initializer(tf.local_variables())
    for epoch_idx in range((previous_step*batch_size)//NUM_TEST_SAMPLE,
    train_config['max_epoch']):
        sess.run(local_variables_initializer)
        start_time = time.time()
        frame_idx_list = np.random.permutation(NUM_TEST_SAMPLE)
        for batch_idx in range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size):
            device_batch_size = batch_size//(COPY_PER_GPU*NUM_GPU)
            
            ##### Next batch indices for preloading the data
            for gi in range(COPY_PER_GPU*NUM_GPU):
                bb = (batch_idx+10)%NUM_TEST_SAMPLE
                nb = frame_idx_list[bb+gi*device_batch_size:bb+(gi+1)*device_batch_size]
                nbb= []
                for _nb in nb:
                    nbb.append(_nb)
                    if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                        nbb.append(_nb-1)
                # This is for Colab not to preload if the RAM is large enough
                if train_config.get('pool_limit',-1)<7000:
                    data_provider.preload(nbb)
            #####
            
            
            total_feed_dict = {}
            for gi in range(COPY_PER_GPU*NUM_GPU):
                batch_frame_idx_list = frame_idx_list[
                    batch_idx+\
                    gi*device_batch_size:batch_idx+(gi+1)*device_batch_size]
                
                batch_data_pc1, batch_data_pc2 = data_provider.provide_batch(batch_frame_idx_list)
                # Get data of main point cloud (pc1)
                input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, encoded_boxes, valid_boxes, keypoint_batch_pc1, \
                    point_batch_pc1 = batch_data_pc1
                
                if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                    # Get data of point clouds of previous frames (pc2)
                    pc2_input_v, pc2_vertex_coord_list, pc2_keypoint_indices_list,\
                        pc2_edges_list, pc2_cls_labels, pc2_encoded_boxes,\
                        pc2_valid_boxes, keypoint_batch_pc2, point_batch_pc2 = batch_data_pc2
                
                
                ### Feed dict for PC1
                t_initial_vertex_features = \
                    input_tensor_sets[gi]['t_initial_vertex_features']
                t_is_training = input_tensor_sets[gi]['t_is_training']
                t_edges_list = input_tensor_sets[gi]['t_edges_list']
                t_keypoint_indices_list = \
                    input_tensor_sets[gi]['t_keypoint_indices_list']
                t_vertex_coord_list = \
                    input_tensor_sets[gi]['t_vertex_coord_list']
                t_n_keypoints = input_tensor_sets[gi]['t_n_keypoints']
                t_n_points = input_tensor_sets[gi]['t_n_points']
                
               
                feed_dict = {
                    t_initial_vertex_features: input_v,
                    t_is_training: True,
                    t_n_keypoints: keypoint_batch_pc1,   
                }
                feed_dict.update(dict(zip(t_edges_list, edges_list)))
                feed_dict.update(
                    dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
                feed_dict.update(
                    dict(zip(t_vertex_coord_list, vertex_coord_list)))
                
                if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                    t_class_labels = input_tensor_sets[gi]['t_class_labels']
                    t_encoded_gt_boxes = input_tensor_sets[gi]['t_encoded_gt_boxes']
                    t_valid_gt_boxes = input_tensor_sets[gi]['t_valid_gt_boxes']
                    
                    feed_dict[t_class_labels] = cls_labels
                    feed_dict[t_encoded_gt_boxes] = encoded_boxes
                    feed_dict[t_valid_gt_boxes] = valid_boxes
                    
                ### Feed dict for PC2
                if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                    pc2_t_initial_vertex_features = \
                        input_tensor_sets[gi]['pc2_t_initial_vertex_features']
                    pc2_t_class_labels = input_tensor_sets[gi]['pc2_t_class_labels']
                    pc2_t_encoded_gt_boxes = input_tensor_sets[gi]['pc2_t_encoded_gt_boxes']
                    pc2_t_valid_gt_boxes = input_tensor_sets[gi]['pc2_t_valid_gt_boxes']
                    pc2_t_edges_list = input_tensor_sets[gi]['pc2_t_edges_list']
                    pc2_t_keypoint_indices_list = \
                        input_tensor_sets[gi]['pc2_t_keypoint_indices_list']
                    pc2_t_vertex_coord_list = \
                        input_tensor_sets[gi]['pc2_t_vertex_coord_list']
                    pc2_t_n_keypoints = input_tensor_sets[gi]['pc2_t_n_keypoints']
                    pc2_t_n_points = input_tensor_sets[gi]['pc2_t_n_points']
                    feed_dict[pc2_t_initial_vertex_features] = pc2_input_v
                    feed_dict[pc2_t_n_keypoints] = keypoint_batch_pc2
                    feed_dict[pc2_t_n_points] = point_batch_pc2
                    # Not necessary to feed in pc2_cls_labels, pc2_encoded_boxes, pc2_valid_boxes
                    # feed_dict = {
                    #     pc2_t_initial_vertex_features: pc2_input_v,
                    #     pc2_t_class_labels: pc2_cls_labels,
                    #     pc2_t_encoded_gt_boxes: pc2_encoded_boxes,
                    #     pc2_t_valid_gt_boxes: pc2_valid_boxes
                    # }
                    feed_dict.update(dict(zip(pc2_t_edges_list, pc2_edges_list)))
                    feed_dict.update(
                        dict(zip(pc2_t_keypoint_indices_list, pc2_keypoint_indices_list)))
                    feed_dict.update(
                        dict(zip(pc2_t_vertex_coord_list, pc2_vertex_coord_list)))
                    feed_dict[t_n_points] = point_batch_pc1
                if TRAIN_OPT == 'flow' and config['flow_parameters']['supervised']:
                    input_tensor_sets[gi]['t_flow_labels']
                    feed_dict[t_flow_labels] = cls_labels
                    
                               
                
                total_feed_dict.update(feed_dict)
            #IPython.embed()
            if train_config.get('is_pseudo_batch', False):
                tf_gradient = [g for g, v in grads_cross_gpu]
                batch_gradient = sess.run(tf_gradient,
                    feed_dict=total_feed_dict)
                batch_gradient_list.append(batch_gradient)
                if batch_ctr % train_config['pseudo_batch_factor'] == 0:
                    batch_gradient_list = list(zip(*batch_gradient_list))
                    batch_gradient = [batch_gradient_list[ggi][0]
                        for ggi in range(len(batch_gradient_list)) ]
                    for ggi in range(len(batch_gradient_list)):
                        for pi in range(1, len(batch_gradient_list[ggi])):
                            batch_gradient[ggi] += batch_gradient_list[ggi][pi]
                    total_feed_dict.update(
                        dict(zip(tf_gradient, batch_gradient)))
                    results = sess.run(train_op, feed_dict=total_feed_dict)
                    batch_gradient_list = []
                batch_ctr += 1
                fetches_pseudo = fetches.copy()
                fetches_pseudo.pop('train_op',None)
                print('!!!! RUN SESSION !!!', 'EPOCH: ', epoch_idx, 'BATCH: ', batch_idx)
                results = sess.run(fetches_pseudo, feed_dict=total_feed_dict)
                if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                    print('flow_loss:',results['flow_loss'])
                if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                    t_cls_loss = t_loss_dict['cls_loss']
                    t_loc_loss = t_loss_dict['loc_loss']
                    t_reg_loss = t_loss_dict['reg_loss']
                    print('cls_loss, loc_loss, reg_loss:',results['cls_loss'],results['loc_loss'],results['reg_loss'])
            else:
                print('!!!! RUN SESSION !!!', 'EPOCH: ', epoch_idx, 'BATCH: ', batch_idx)
                results = sess.run(fetches, feed_dict=total_feed_dict)
                if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
                    t_cls_loss = t_loss_dict['cls_loss']
                    t_loc_loss = t_loss_dict['loc_loss']
                    t_reg_loss = t_loss_dict['reg_loss']
                    print('cls_loss, loc_loss, reg_loss:',results['cls_loss'],results['loc_loss'],results['reg_loss'])
                if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
                    print('flow_loss:',results['flow_loss'])
            #print('Check if returns correct things for pc2')
            #IPython.embed()
            # TODO: Save name should be changed according to TRAIN_OPT
            #IPython.embed()
            if 'max_steps' in train_config and train_config['max_steps'] > 0:
                if results['step'] >= train_config['max_steps']:
                    checkpoint_path = os.path.join(train_config['train_dir'],
                        train_config['checkpoint_path'])
                    config_path = os.path.join(train_config['train_dir'],
                        train_config['config_path'])
                    train_config_path = os.path.join(train_config['train_dir'],
                        'train_config')
                    print('save checkpoint at step %d to %s'
                        % (results['step'], checkpoint_path))
                    saver.save(sess, checkpoint_path,
                        latest_filename='checkpoint',
                        global_step=results['step'])
                    save_config(config_path, config_complete)
                    save_train_config(train_config_path, train_config)
                    raise SystemExit
        print('STEP: %d, epoch_idx: %d, lr: %f, time cost: %f'
            % (results['step'], epoch_idx, results['learning_rate'],
            time.time()-start_time))
        if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
            print('cls:%f, loc:%f, reg:%f, loss: %f'
                % (results['cls_loss'], results['loc_loss'], results['reg_loss'],
                results['total_loss']))
        if TRAIN_OPT == 'flow' or TRAIN_OPT == 'both':
            print('flow:%f' % (results['flow_loss']))
        
        if TRAIN_OPT == '3Ddet' or TRAIN_OPT == 'both':
            for class_idx in range(NUM_CLASSES):
                print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                    % (class_idx,
                    results['recall_%d'%class_idx],
                    results['precision_%d'%class_idx],
                    results['mAP_%d'%class_idx],
                    results['loc_loss_cls_%d'%class_idx]))
                print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
                %(
                results['loc_loss_cls_%d_box_%d'%(class_idx, 0)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 1)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 2)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 3)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 4)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 5)],
                results['loc_loss_cls_%d_box_%d'%(class_idx, 6)]),
                )

        # add summaries ====================================================
        for key in metrics_update_ops:
            write_summary_scale(key, results[key], results['step'],
                train_config['train_dir'])
        write_summary_scale('learning rate', results['learning_rate'],
            results['step'], train_config['train_dir'])
        
        # Eval =============================================================
        if FLOW_VAL_DATASET_SPLIT_FILE is not None and (epoch_idx + 1) % train_config['validate_every_epoch'] == 0:
            eval_total_loss = eval_one_step()
        
        # save checkpoint ==================================================
        # TODO: Update the save name according to TRAIN_OPT
        if (epoch_idx + 1) % train_config['save_every_epoch'] == 0:
            checkpoint_path = os.path.join(train_config['train_dir'],
                train_config['checkpoint_path'])
            config_path = os.path.join(train_config['train_dir'],
                train_config['config_path'])
            train_config_path = os.path.join(train_config['train_dir'],
                'train_config')
            print('save checkpoint at step %d to %s'
                % (epoch_idx, checkpoint_path))
            saver.save(sess, checkpoint_path,
                latest_filename='checkpoint',
                global_step=results['step'])
            save_config(config_path, config_complete)
            save_train_config(train_config_path, train_config)
        # =====================================================================    
        if FLOW_VAL_DATASET_SPLIT_FILE is not None and (epoch_idx + 1) % train_config['validate_every_epoch'] == 0 and eval_total_loss < min_eval_loss:
            min_eval_loss = eval_total_loss
            checkpoint_path_best = os.path.join(train_config['train_dir'],'best',
                train_config['checkpoint_path'])
            write_min_loss(os.path.join(train_config['train_dir'],'best','min_eval_loss'),eval_total_loss)
            copyfile(checkpoint_path+"-{}.index".format(results['step']),checkpoint_path_best+".index")
            copyfile(checkpoint_path+"-{}.meta".format(results['step']),checkpoint_path_best+".meta")
            copyfile(checkpoint_path+"-{}.data-00000-of-00001".format(results['step']),checkpoint_path_best+".data-00000-of-00001")
            #saver.save(sess, checkpoint_path)
    # save final
    # TODO: Update the save name according to TRAIN_OPT
    checkpoint_path = os.path.join(train_config['train_dir'],
        train_config['checkpoint_path'])
    config_path = os.path.join(train_config['train_dir'],
        train_config['config_path'])
    train_config_path = os.path.join(train_config['train_dir'],
        'train_config')
    saver.save(sess, checkpoint_path,
        latest_filename='checkpoint',
        global_step=results['step'])
    save_config(config_path, config_complete)
    save_train_config(train_config_path, train_config)


