#/bin/bash

# DETECTION
FOLDER_NAME=car_detection_after_flow_step4
python3 run.py /trainings/checkpoints/$FOLDER_NAME/ --dataset_root_dir /kitti_root_3d/pointgnn/ --output_dir /trainings/checkpoints/$FOLDER_NAME/inference

# EVALUATION
./kitti_native_evaluation/evaluate_object_3d_offline /kitti_root_3d/object_detection_2012/labels/training/label_2/ /trainings/checkpoints/$FOLDER_NAME/inference




