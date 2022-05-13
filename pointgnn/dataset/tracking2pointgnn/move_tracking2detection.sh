#!/bin/bash

# Move all the data in KITTI Tracking dataset into KITTI Detection folder format
# --training_drives: Drive IDs to be converted and used for training. Use only training_drives if split is testing. (Ex: 11 15 16 18)
# --validation_drives: Drive IDs to be converted and used for validation. Use only training_drives if split is testing. (Ex: 11 15 16 18)
# --tracking_path: Path where the drive label folders are kept. Should contain folders 0011, 0015, 0016, 0018 etc. as drive IDs given in the example. The labels should be in KITTI Detection format. Use combine_drive_labels.py to convert.
# --output_path: Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, the same folders are generated under <tracking_path>/drives_in_kitti.
# --split: training or testing split in KITTI
# --remove_first: To skip the first frame of drives in the train.txt and val.txt files

## To train pointgnn flow with tracking dataset					
python move_tracking2detection.py 	--training_drives 0 1 2 3 4 5 6 7 8 9 10 12 13 14 17 19 20 \
					--validation_drives 11 15 16 18 \
					--root_dir /kitti_root_tracking/drives_in_kitti \
					--tracking_path /kitti_root_tracking \
					--output_path /kitti_root_tracking/detection_format/pointgnn \
					--split training \
					--remove_first
