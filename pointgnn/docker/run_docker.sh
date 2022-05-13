#!/bin/bash


# Path where the KITTI object detection dataset for 3D detection is kept
kitti_root_3d="kitti_object_detection"
# Path where the KITTI tracking dataset is kept
kitti_root_tracking="tracking_dataset"
# Path to the codes in the host machine
point_gnn="cvpr_pointgnn"
# Where the logs and checkpoints will be kept in the host
trainings="/trainings"

# Run docker
docker run -it --gpus all  --rm -v $kitti_root_3d:/kitti_root_3d \
				-v $kitti_root_tracking:/kitti_root_tracking \
				-v $point_gnn:/point_gnn \
				-v $trainings:/trainings \
				selfsup_pointgnn_flow

