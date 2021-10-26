# You Only Look Once: Unified, Real-Time Object Detection

Coder: Jay Liao, jl64465

This directory contains the code for the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640).

I used and worked from 3 open-source PyTorch implementation of YOLO:
* [pytorch-yolo-v1](https://github.com/zzzheng/pytorch-yolo-v1)
* [YOLOv1-Pytorch](https://github.com/GitHberChen/YOLOv1-Pytorch)
* [pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1)

## Requirements
* Python 3.6.13
* PyTorch 1.9.1
* JupyterLab 3.1.13
* Scikit-Image 0.17.2
* Notebook 6.4.4
* Matplotlib 3.3.4
* torchinfo 1.5.3
* tqdm 4.62.3
* opencv2

## Commands
For training:
```
python3 -m train \
    --name voc2007_resnet50 \
    --backbone resnet50 \
    --dataset voc2007 \
    --batch_size 16 \
    --warm_up_epochs 3
```
This corresponds to using a ResNet50 as the backbone CNN for YOLO. We will train on the Pascal VOC2007 dataset with batch sizes of 16. The model will linearly increase the learning rate by a factor of 10 for 3 epochs.

For testing:
```
python3 -m test \
    --set test \
    --year 2007 \
    --backbone resnet50 \
    --step 10990
```
This corresponds to using the trained ResNet50 backbone YOLO for inference. We will do inference on the Pascal VOC2007 test set after the model has been trained for 10990 steps.
