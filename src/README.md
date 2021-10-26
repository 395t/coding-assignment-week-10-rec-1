# RetinaNet - Focal Loss for Dense Object Detection

Coder: Elias Lampietti, ejl2425

Implementation of Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár's [paper](https://arxiv.org/pdf/1708.02002.pdf) from 2017

This code was adapted from the RetinaNet model implemented in the Facebook AI Research library [Detectron2](https://github.com/facebookresearch/detectron2)

## RetinaNet Model

The RetinaNet is a one-stage model used for object detection.
This performs much faster than two-stage models such as Mask-RCNN and Faster-RCNN as RetinaNet has a combined architecture for both detection and classification.
The Focal Loss introduced in the RetinaNet model mitigates the accuracy issue caused in one-stage models by background objects being easier to detect than foreground objects.

## Dataset

We used the PASCAL VOC [2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html), [2008](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html), and [2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) datasets to fine-tune and evaluate the RetinaNet
The PASCAL VOC dataset provides standardised image data sets for object class recognition with annotations for each image.

## Data Prerequisites

Since the Detectron2 models take input data in the COCO json format, we used [roboflow](https://roboflow.com/) to convert the PASCAL VOC data into the COCO json format and make it available for public download with the following commands.

#### 2007 train/validation: 
!curl -L "https://app.roboflow.com/ds/ji2cS6UUK4?key=SCGzhDvz6i" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#### 2007 test: 
!curl -L "https://app.roboflow.com/ds/BfERhbEO1E?key=82JnLtE0Z5" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#### 2008 train/validation/test : 
!curl -L "https://app.roboflow.com/ds/BFV0OeJdj5?key=jeeyAa5YT9" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

#### 2012 train/validation/test: 
!curl -L "https://app.roboflow.com/ds/C6enLy92Ft?key=Mzi73TKWJ3" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip



## References

