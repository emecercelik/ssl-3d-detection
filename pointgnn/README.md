# Multi-frame self-supervised flow-based Point-GNN
This repository contains the implementation of the paper "3D Object Detection with a Self-supervised Lidar Scene Flow Backbone". We mainly use KITTI Tracking Dataset for the scene flow task (no annotations needed) and KITTI Object Detection dataset for the 3D detection task. Please see below for arranging the datasets.

We provide the configuration files under `configs` folder for our main results. Please refer to `train.sh` script for starting trainings for different experiments. For inference and evaluation results, please refer to the `test_on_val.sh` script. 

Our checkpoints for the results of main experiments are given under `checkpoints` folder, which has the same structure with the `configs` folder.

You can refer to the `Prerequisites` section below for preparing the environment to run the training or check the `docker` folder for the `Dockerfile`, building the docker image (`build_image.sh`) and starting the container (`run_docker.sh`). 

## KITTI Tracking Dataset Preparation

### Downloading the dataset
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

### Converting to a consumable format

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

After this conversion, arranging the data into the oroginal Point-GNN data structure is recommented:
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

## KITTI 3D Object Detection Dataset Preparation
After downloading the KITTI 3D Object Detection Dataset, please arrange the data into the following structure.

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
        

## Config explanations
There are 2 files for configuration parameters. `*_train_config` is mainly for the architecture parameters of Point-GNN and `*_train_train_config` is mainly for the training parameters. Scene flow related training parameters and restoring from backbones are also defined in `*_train_config` file. Below is the explanation for only parameters that are added for these experiments. The rest is original Point-GNN parameters.

In `*_train_config`, the extended parameters can be seen below:

```json
"flow_parameters": {
    	"num_flow_points": 1024,
    	"flow_loss_weight": 1.0,
    	"train":"3Ddet",
    	"restore_path": "/trainings/checkpoints/car_flow_selfsup_s4/model-401340",
    	"supervised": false,
    	"flow_supervised_path": "/kitti_root_tracking/supervised_flow/",
    	"filter_layers": false,
    	"eval_dataset": "val.txt",
    	"repeat_batch": 1,
    	"backbone_restore": "/trainings/checkpoints/car_3Ddet_s3/best/model",
    	"detection_head_restore": "/trainings/checkpoints/car_3Ddet_s3/best/model",
    	"scene_flow_head_restore": "/trainings/checkpoints/car_flow_selfsup_s4/best/model",
        "alternating_train": false
    	
```

Their meanings are given below:

* `num_flow_points`: Number of points that will be taken from each frame's lidar point cloud for the scene flow training. These correspond to the sampled points in Fig.2 of the paper. These are the lidar points sampled from the whole point cloud of frames t and t+1. Features of the sampled points are obtained by interpolation from the Point-GNN keypoint features.

* `flow_loss_weight`: Loss weight for scene flow cycle loss. This is useful while training the network with both scene flow and 3D detection heads. The default loss weight of the total 3D detection loss is 1. Therefore, this should be adjusted while training both heads.

* `train`: Indicates which head will be trained. For 3D detection head it is "3Ddet" and  for scene flow it is "flow". 

* `restore_path`: This shows the path to the checkpoint, where the scene flow head will be initialized for the first time training. Scene flow head's architecture is similar to the FlowNet3D architecture. The point feature learning "set conv layers" of FlowNet3D are changed with the Point-GNN backbone. The rest of the architecture is kept the same. To be able to train the FlowNet3D with the self-supervised loss, we initialize weights with a pre-trained model on a simulated data. We use this pre-trained model's weights, where possible, to initialize our FlowNet3D model to have a stable trianing. Use only for Step 1 of experiments. The pre-trained model can be downloaded from the repository of ["Just Go with the Flow: Self-Supervised Scene Flow Estimation"](https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation) published in CVPR 2020. Please use `flow/old2new_ckpt.py` to convert the checkpoint variable names into the supported format.

* `filter_layers`: This is to filter out weights of some layers while restoring from the model checkpoint given with restore_path. Because some layers of FlowNet3D are not used in Flow-PointGNN. 

* `eval_dataset`: A txt split file that indicates frame ids in the dataset that will be used for evaluation. This should be kept in 3DOP_splits folder of the dataset as indicated in the original Point-GNN README below. 

* `repeat_batch`: This parameter shows how many times a set of points will be sampled from the same frame while training for the scene flow. This indicates an internal batch size for scene flow training that uses the same keypoint features. 

* `backbone_restore`: The model checkpoint path to restore the backbone layer weights. 

* `detection_head_restore`: The model checkpoint path to restore the 3D detection head layer weights.

* `scene_flow_head_restore`: The model checkpoint path to restore the scene flow head layer weights. 


In `*_train_train_config`, some of the parameters important for training are explained below:

* `NUM_GPU`: Number of GPUs to be used during training.
* `batch_size`: The total batch size. If `NUM_GPU=2` and `batch_size=4`, each GPU will have `batch_size=2`. 
* `decay_factor`:  Decay factor of the learning rate every `decay_step` number of steps. 
* `decay_step`: Number of steps to apply a learning rate decay. Each mini batch is counted as a step. So in one epoch, number of steps depends on the `batch_size`.
* `initial_lr`: Initial learning rate that the training will be started. According to the number of steps, this is decayed. A training can be started, where it left off. This is done automatically from the last checkpoint in the `train_dir`, if any. This means, continueing the training means also the learning rate takes the decayed value of that step, where the training continues. 
* `save_every_epoch`: The checkpoints will be saved once in every defined epochs. 
* `train_dataset`: Train split as a txt file, which indicates frame ids to be used during training. 
* `train_dir`: Log directory, where the checkpoints and tensorboard summaries will be saved.
* `pool_limit`: Indicates how many frames will be kept in the RAM for each of training and validation data. If `1000`, in total `2000` frames are kept in the RAM (`1000` for training and `1000` for validation). This value should be adjusted according to the RAM of the computer. Reading data is faster when the number is higher, but it requires more RAM. 


### Prerequisites
It is possible to use the environment for the Point-GNN work, as explained in the original repository:

Tensorflow 1.15 is used for this implementation. Please [install CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) if you want GPU support.   
```
pip3 install --user tensorflow-gpu==1.15.0
```

To install other dependencies: 
```
pip3 install --user opencv-python
pip3 install --user open3d-python==0.7.0.0
pip3 install --user scikit-learn
pip3 install --user tqdm
pip3 install --user shapely
```


## Checkpoints

Checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1DM4PumzxGblidBHXC8ao7yP3GHUl8KjC?usp=sharing).