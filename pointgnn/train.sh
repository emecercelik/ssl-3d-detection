#/bin/bash

### Baseline 3D detection training
python3 train.py configs/detection_baseline/car_auto_T3_train_train_config configs/detection_baseline/car_auto_T3_train_config --dataset_root_dir /kitti_root_3d/pointgnn/ 


### Flow training Step 1
python3 train.py configs/selfsupervised_flow_step1/car_auto_T3_train_train_config configs/selfsupervised_flow_step1/car_auto_T3_train_train_config --dataset_root_dir /kitti_root_tracking/detection_format/pointgnn/

### Detection training (backbone initialization from self-supervised flow ckpt) Step 2
python3 train.py configs/detection_after_flow_step2/car_auto_T3_train_train_config configs/detection_after_flow_step2/car_auto_T3_train_config --dataset_root_dir /kitti_root_3d/pointgnn/ 

### Flow training Step 3
python3 train.py configs/selfsupervised_flow_step3/car_auto_T3_train_train_config configs/selfsupervised_flow_step3/car_auto_T3_train_train_config --dataset_root_dir /kitti_root_tracking/detection_format/pointgnn/

### Detection training (backbone initialization from self-supervised flow ckpt) Step 4
python3 train.py configs/detection_after_flow_step4/car_auto_T3_train_train_config configs/detection_after_flow_step4/car_auto_T3_train_config --dataset_root_dir /kitti_root_3d/pointgnn/ 
