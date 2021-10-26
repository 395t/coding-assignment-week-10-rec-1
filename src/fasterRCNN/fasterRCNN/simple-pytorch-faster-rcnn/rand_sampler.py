from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import random
import ipdb

f=open('/home/sritank/Documents/Acads/Fall 2021/CS395T Deep Learning Seminar/fasterRCNN/VOCdevkit2008/VOC2008/ImageSets/Main/val (copy).txt','r')
x=f.readlines()
test=[]
val=[]
indices=[]
# ipdb.set_trace()
for i in range(0,len(x)):
    x[i]=x[i].strip('\n')

# for i in range(len(x)):
#     r=random.randint(0,len(x)-1)
#     # ipdb.set_trace()
#     if(len(test)<1000 and r<0.5):
#         test.append(x[i])
#     else:
#         val.append(x[i])

while(len(indices)<1000):
    r=random.randint(0,len(x)-1)
    # ipdb.set_trace()
    if(r not in indices):
        indices.append(r)

    

for i in range(0,len(x)):
    if(i in indices):
        test.append(x[i])
    else:
        val.append(x[i])

testfile = open('/home/sritank/Documents/Acads/Fall 2021/CS395T Deep Learning Seminar/fasterRCNN/VOCdevkit2008/VOC2008/ImageSets/Main/test_sampled.txt', 'w')
testfile.write('\n'.join(test))
testfile.close()

valfile = open('/home/sritank/Documents/Acads/Fall 2021/CS395T Deep Learning Seminar/fasterRCNN/VOCdevkit2008/VOC2008/ImageSets/Main/val_sampled.txt', 'w')
valfile.write('\n'.join(val))
valfile.close()