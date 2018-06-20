# Light-Weight Multi-View 3D Object Detection

## Requirements
* Python 3.5.2 or higher
* TensorFlow-GPU 1.3.0 or higher
    * CUDA
    * cuDNN
    * GPU

## Build
First, change `-arch=sm_52` in the files `lib/setup.py` and `lib/roi_pooling_layer/make.sh` to an appropriate value for your GPU.
```bash
./make.sh
```

## Dataset
Download left color images, Velodyne point clouds, camera calibration matrices, and training labels
from `http://www.cvlibs.net/datasets/kitti/eval_object.php`.
Split them into a training set and a validation set (e.g. according to `Train/Val Split` in `http://www.cs.toronto.edu/objprop3d/downloads.php`)
and specify the directories of them by `DIR_TRAIN` and `DIR_VAL` in `train.py`.

## Train
Please read help by `python train.py -h`.

## Infer
Please read help by `python infer.py -h`.

## Evaluate
```bash
cd path/to/inference/output_dir
mkdir data
mv *.txt data/

cd anywhere/you/like
git clone https://github.com/prclibo/kitti_eval.git
cd kitti_eval
c++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
./evaluate_object_3d_offline path/to/validation/labels path/to/inference/output_dir
```
