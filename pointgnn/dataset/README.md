# Preparing KITTI Tracking Dataset for Self-supervised Scene Flow Training

## Downloading the dataset
The KITTI Multi-object Tracking Dataset (Tracking dataset) can be downloaded using the links in the official [website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) of KITTI. To use with the extended Frustum PointNets module, the [left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), [Velodyne point clouds](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [camera calibration matrices](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip), and [training labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) should be downloaded.

Downloaded zip files should be extracted in the same root_dir to have the following path order (0000,0001,... are drive indices and 000000,000001,... are frame indices):

```
root_dir
	data_tracking_image_2
		training
			image_02
				0000
					000000.png
					...
				0001
					000000.png
					...
				...
		testing
			image_02
				...
	data_tracking_label_2
		training
			label_02
				0000.txt
				0001.txt
				...
	data_tracking_velodyne
		training
			velodyne
				0000
					000000.bin
					...
				0001
					000000.bin
					...
				...
		testing
			velodyne
				...
	data_tracking_calib
		training
			calib
				0000.txt
				0001.txt
				...
		testing
			calib
				...
		
```

## Converting to a consumable format

After download, the KITTI tracking dataset should be converted into a consumable format. The `dataset/tracking2pointgnn/tracking2object.sh` script will convert the labels into the KITTI object detection dataset format. Afterwards, the `dataset/tracking2pointgnn/move_tracking2detection.sh` script copies all the data in a new directory in the KITTI object detection dataset format as seen below:

```
root_dir
	training
		calib
			000000.txt
			000001.txt
			...
		image_2
			000000.png
			000001.png
			...
		label_2
			000000.txt
			000001.txt
			...
		velodyne
			000000.bin
			000001.bin
			...
	testing
		calib
			000000.txt
			000001.txt
			...
		image_2
			000000.png
			000001.png
			...
		velodyne
			000000.bin
			000001.bin
			...
```

After this conversion, please follow the original Point-GNN instructions to arrange the data into the Point-GNN data structure:
    DATASET_ROOT_DIR
    ├── image                    #  Left color images
    │   ├── training
    |   |   └── image_2            
    │   └── testing
    |       └── image_2 
    ├── velodyne                 # Velodyne point cloud files
    │   ├── training
    |   |   └── velodyne            
    │   └── testing
    |       └── velodyne 
    ├── calib                    # Calibration files
    │   ├── training
    |   |   └──calib            
    │   └── testing
    |       └── calib 
    ├── labels                   # Training labels
    │   └── training
    |       └── label_2
    └── 3DOP_splits              # split files.
        ├── train.txt
        ├── train_car.txt
        └── ...
        
The `dataset/tracking2pointgnn/move_tracking2detection.sh` also creates the split `*.txt` files, which can be copied under `3DOP_splits` folder to use. In order to use the KITTI tracking dataset test split lidar point clouds for the self-supervised scene flow training, you can follow the same steps. The drive IDs shouldn't overlap with the train split drive IDs. 
