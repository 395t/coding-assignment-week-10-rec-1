## Dataset 
The following image classification datasets were used:
-PASCAL VOC 2007 
-PASCAL VOC 2008 
-PASCAL VOC 2012
-COCO 2014

Here are some example annotated images from the training dataset:

![training data](https://user-images.githubusercontent.com/7085644/138916591-9f9ac18b-caa0-44b4-9c03-ff5cffdaa76a.PNG)

It should be noted here that PASCAL VOC 2008 contains ~700 images from the 2007 dataset. Similarly, the VOC 2012 dataset contains training examples from the VOC 2007-2011 datsets. Since the goal is 2D recognition, the segmentation task is not studied here.

## Hardware
All experiments on the VOC dataset were performed on Colab, the UT CS cluster and a lab desktop with Intel I9-10900KF and NVIDIA RTX 3080 GPU.


## 2D image recognition using fasterRCNN architecture

## Training

For all the VOC experiments, the fasterRCNN used a vgg16 as the fastRCNN model. For the COCO experiment, the vgg16 model made the training unstable and the losses diverged to NaN's, low learning rates of (0.00001) were stable but not practical for learning. 
![vgg nan](./figures/vgg16_nan.png)

Resnet101 was found to do better and was stable during the training process. Due to memory limitations on my desktop, the batch size for all datasets is 1, i.e. BATCH_SIZE=1

Resnet101 was used for the COCO experiments alone. Training on COCO 2014 was really slow (~11 hrs) per epoch (160k iterations) on my machine. So the training was truncated at 100k iterations.

The region proposal network (RPN) is a CNN that is also trained to learn the proper region proposals. The 4 step alternating training strategy is used to train the model here. A stochastic gradient descent (SGD) with learning rate (lr) of 0.001 was used for all results shown here.

![voc2007 train](./figures/2007_train.png)

![voc2012 train](./figures/2012_train.png)

![coco2014 train](./figures/coco2014_total_loss.png)

![voc2014 train2](./figures/coco2014_rpn_loss.png)

## Testing

The test dataset for VOC 2007 was readily available and the mAP could be calculated over the given set. However for the VOC 2008 and 2012 datasets, the test dataset was missing annotations to seveal test images. To resolve this we sampled 1000 random samples from the validaton set and used as testing for VOC 2008 and VOC 2012.

![voc2007 train](./figures/2007_test.png)

![voc2012 train](./figures/2012_test.png)

For COCO 2014, the 5000 test images were used. Since each epoch has 164k iterations, doing only 100k iterations doesn't provide a great mAP score, hence it is not reported here.


Demonstration of how the network learning evolves

![epoch1](./figures/output_1.png)
![epoch1](./figures/output_2.png)
![epoch1](./figures/output_3.png)
![epoch1](./figures/output_4.png)
![epoch1](./figures/output_fin.png)


## Summary

The network was successfully trained on VOC 2007 and 2012 datasets. Due to hardware constraints, the COCO dataset training couldn't be completed, but given enough time the plot trends showed convergence and that the model was learning. Pre-training greatly reduces the training time in all cases. 


# RetinaNet - Focal Loss for Dense Object Detection

Coder: Elias Lampietti, ejl2425

Implementation of Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár's [paper](https://arxiv.org/pdf/1708.02002.pdf) from 2017

This code was adapted from the RetinaNet model implemented in the Facebook AI Research library [Detectron2](https://github.com/facebookresearch/detectron2)

Notebooks that walk through this training/fine-tuning/evaluation process for the VOC 2007,2008,2012 datasets can be found [here](https://github.com/395t/coding-assignment-week-10-rec-1/tree/main/notebooks/RetinaNet)

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

## Training

We started with the pre-trained RetinaNet model from Detectron2 and then customized it by training and fine-tuning this model for the VOC 2007, 2008, and 2012 datasets.

## Results

The total loss during training for each dataset is shown below:

![VOC 2007 Training Loss](https://user-images.githubusercontent.com/7085644/138904733-2efc6ed0-f2bd-4903-94e9-3431388777fa.PNG)
![VOC 2008 Training Loss](https://user-images.githubusercontent.com/7085644/138904433-3646b9e0-65a1-49f5-b40e-acfcf3132e01.PNG)
![VOC 2012 Training Loss](https://user-images.githubusercontent.com/7085644/138904063-635d2ee0-43ae-4d5a-a0fa-b755d60909d7.PNG)

These charts show that the model trains well and is able to reduce the total loss to around 0.3 for all three datasets after 1.5k iterations.

The following image displays the changes in the model's LR during training caused by a warmup LR scheduler.

![2012 LR](https://user-images.githubusercontent.com/7085644/138907869-c966f1bf-afcf-4fb1-9401-1e76cde0990f.PNG)

The average precision (AP) for a few of the 20 classification categories is show below:

#### 2007:

![2007 Training AP](https://user-images.githubusercontent.com/7085644/138908551-0487b38a-fb67-45d7-a546-3e781bbe2d6a.PNG)

#### 2008:

![2008 Training AP Objects](https://user-images.githubusercontent.com/7085644/138908614-d0286440-cf66-4942-9efd-b50b343ca50d.PNG)

#### 2012:

![2012 Training AP](https://user-images.githubusercontent.com/7085644/138908599-48e125e0-2846-4fff-8fe9-537effafa2cb.PNG)

These plots show that the model struggles on detecting the bottle object on the 2007 and 2008 datasets which are similar, however it is able to detect the bottle much better on the 2012 dataset.
These also show that all models are able to predict the boat object relatively well and the bus object very well with APs up to almost 70.

The following charts are the mean average precision (mAP) during training.

#### 2007:

![2007 Training AP](https://user-images.githubusercontent.com/7085644/138925794-58c9d04e-3888-42b5-a234-1a8263808463.PNG)

#### 2008:

![2008 training ap](https://user-images.githubusercontent.com/7085644/138925810-a8f64fdc-64d2-45fd-9176-ab07bf304c26.PNG)

#### 2012:

![2012 Training AP](https://user-images.githubusercontent.com/7085644/138925825-2d29f92b-11f3-481c-8e07-e76774f39409.PNG)

Although each dataset shows increasing mAP, the 2007 and 2012 datasets show the greatest increase with a final mAP of around 45.

#### Evaluation

After training and fine-tuning the model on the 3 datasets, we evaluated them with the mean average precision metric.
Below is a table showing how each model performed based on mean average precision for object detection of the 20 different objects.

![RetinaNet MAP](https://user-images.githubusercontent.com/7085644/138915892-3f4fa337-aba6-42d9-8663-ac473bcfa767.PNG)

These results show that the RetinaNet performed the best on the PASCAL VOC 2007 dataset achieving a mean average precision of 46.21.
The reason for the lower result of 31.24 in the 2008 dataset even though it is similar to the 2007 is because the provided annotations had some strange values mixed in such as "personhandheadhandfootfoot" that the model was trying to classify. 
The RetinaNet was also able to achieve good results for the PASCAL VOC 2012 dataset with a mean average precision of 43.78.

The following images show how well the RetinaNet is able to perform object detection for each VOC dataset with random test images.

#### 2007:

![2007 test images](https://user-images.githubusercontent.com/7085644/138924638-bb3983ba-087c-4a6e-b3f0-e340cc3991d4.PNG)

#### 2008:

![2008 test images](https://user-images.githubusercontent.com/7085644/138924653-dc899a06-eb37-43eb-9f6d-9aee947bcb07.PNG)

#### 2012:

![2012 test images](https://user-images.githubusercontent.com/7085644/138924662-44e671cb-07ad-4c5e-b032-a82b5f71be35.PNG)

These results show that RetinaNet is able to classify vehicles better than animals as it achieves 93-94% certainty for the bus on each dataset however it fluctuates from 66% to 83% certainty for the dog with 2007 being the worst and 2008 performing the best and 2012 being average at 75% certainty.

## References

[Detectron2: Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick, 2019](https://github.com/facebookresearch/detectron2)
[Roboflow data loading and training tutorial: Jacob Solawetz, 2020](https://blog.roboflow.com/how-to-train-detectron2/)
