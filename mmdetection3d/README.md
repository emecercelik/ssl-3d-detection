# Self-supervised pre-trainining PointPillars

## Description

This README file is about how to install and use our self-supervised pre-training approach for PointPillars in mmdetection3d repository.

## Prerequirements

We test our approach based on the following setting:

```
CUDA=11.0
pytorch==1.7.0
mmcv-full==1.4.0
mmdet==2.11.0
mmdet3d==0.13.0
mmsegmentation==0.13.0
```

## Installation

For the model can be successfully used, please according to the followed steps.
1. Install the official [mmdetetcion3d](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md)
2. We use KITTI and nuScenes datasets for our training. Please arrange the dataset structure and pre-precess following [data_preparation.md](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/docs/en/data_preparation.md)
3. Install `kaolin` in mmdetection3d folder
```
cd mmdetection3d
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout v0.1
python setup.py install
```
3. Install `torch_geometric`

```
pip install torch_geometric==1.7.2
pip install torch-sparse==latest+{cu_version} -f https://pytorch-geometric.com/whl/{torch_version}.html
pip install torch-scatter==latest+{cu_version} -f https://pytorch-geometric.com/whl/{torch_version}.html
pip install torch-cluster==latest+{cu_version} -f https://pytorch-geometric.com/whl/{torch_version}.html
```
Please replace `{cu_version}` and `{torch_version}` in the url to your desired one.
For example,
```
pip install torch-sparse==latest+cu110 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
```

4. Replace folders 

a. In path `/mmdetection3d`, replace these official folders with ours:
```
configs, mmdet3d, tools
```
b. In path `/opt/conda/lib/python3.7/site-packages/`, replace the official `mmdet` folder with ours

5. Add our `Checkpoints` folder into path `/mmdetection3d`

## Dataset

We use KITTI and nuScenes datasets for our training. Please arrange the dataset structure following [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/docs/en/data_preparation.md) 

## Usage
### KITTI
1. **PointPillars**

`sh ./tools/test_our_pointpillars_kitti.sh`

### nuScenes

1. **PointPillars**

`sh ./tools/test_our_pointpillars_nus.sh`

2. **CenterPoint**

`sh ./tools/test_our_centerpoint_nus.sh`

3. **SSN**
`sh ./tools/test_our_ssn_nus.sh`

## Checkpoints for Detection Task
|             | 1% |2.5%|5%|10%|20%|100%|
|-------------|:--:|:--:|:--:|:---:|:---:|:---:|
|PointPillars | ✗         | ✓            | ✓           | ✓      | ✓   | ✓    |
|CenterPoint. | ✗         | ✓            | ✓           | ✓      | ✗   | ✓    |
|SSN          | ✓         | ✓            | ✓           | ✓      | ✗   | ✗    |


(https://drive.google.com/drive/folders/1DM4PumzxGblidBHXC8ao7yP3GHUl8KjC?usp=sharing).
