#!/bin/bash

#Converts KITTI tracking label format into KITTI object detection format.
#    tracking_path: Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders
#    drive_ids : List of drive ids whose labels will be converted. [0,1,2,3]
#    output_path : Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, see below.
    
#    Generates a new folder named as "drives_in_kitti" in the tracking_path if the output_path is None. Inside this folder, drive labels are separated according to their names.
#    tracking_path
#    ---drives_in_kitti
#       ---0000
#          ---000000.txt
#          ---000001.txt
#          ---*.txt
#       0000_val.txt : A text file with the frame numbers of the labels in 0000
#       ---0011
#          ---000000.txt
#          ---000001.txt
#          ---*.txt
#       0011_val.txt : A text file with the frame numbers of the labels in 0011


# to convert KITTI tracking dataset format into KITTI object detection format
python tracking2object.py 	--tracking_path /kitti_root_tracking/ \
				--drive_ids 0 1 2 3 4 5 6 7 8 9 10 12 13 14 17 19 20 \
				--output_path /kitti_root_tracking/drives_in_kitti 

