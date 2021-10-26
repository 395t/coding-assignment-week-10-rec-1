"""
Modified from https://github.com/GitHberChen/YOLOv1-Pytorch/blob/master/src/model.py
"""
from typing import Tuple, Type
from enum import Enum
import torchvision
import torch
from torch import nn
from typing import Tuple, List, Optional, Union
from torch import nn
from torchvision import models as Models
from os import path as osp
from math import sqrt
import os
from config import *


class CNNBlock(nn.Module):
    """
    From https://medium.com/mlearning-ai/object-detection-explained-yolo-v1-fb4bcd3d87a1
    """
    def __init__(self,in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.l_relu = nn.LeakyReLU(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        return self.l_relu(self.bn(self.conv(x)))


class Backbone(nn.Module):
    def __init__(self, backbone_name: str):
        super(Backbone, self).__init__()
        pretrained_models = {
            'resnet18': (Models.resnet18(True), nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 5),
            'resnet50': (Models.resnet50(True), nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1), 5),
            'vgg11': (Models.vgg11(True), nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 6),
            'vgg16': (Models.vgg16(True), nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), 6),
        }

        if backbone_name == "yolo":
            """
            From https://medium.com/mlearning-ai/object-detection-explained-yolo-v1-fb4bcd3d87a1
            """
            layers = [[], [], [], [], [], []]
            layers[0].append(CNNBlock(3, 64, kernel_size=7, stride=2, padding=3))  #1
            layers[0].append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers[1].append(CNNBlock(64, 192, kernel_size=3, padding=1))  #2
            layers[1].append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers[2].append(CNNBlock(192, 128, kernel_size=1))  #3
            layers[2].append(CNNBlock(128, 256, kernel_size=3, padding=1))
            layers[2].append(CNNBlock(256, 256, kernel_size=1))
            layers[2].append(CNNBlock(256, 512, kernel_size=3, padding=1))
            layers[2].append(nn.MaxPool2d(kernel_size=2, stride=2))
            for _ in range(4):  # 4
                layers[3].append(CNNBlock(512, 256, kernel_size=1))
                layers[3].append(CNNBlock(256, 512, kernel_size=3, padding=1))
            layers[3].append(CNNBlock(512, 512, kernel_size=1))
            layers[3].append(CNNBlock(512, 1024, kernel_size=3, padding=1))
            layers[3].append(nn.MaxPool2d(kernel_size=2, stride=2))
            for _ in range(2):  # 5
                layers[4].append(CNNBlock(1024, 512, kernel_size=1))
                layers[4].append(CNNBlock(512, 1024, kernel_size=3, padding=1))
            layers[4].append(CNNBlock(1024, 1024, kernel_size=3, padding=1))
            layers[4].append(CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1))
            layers[5].append(CNNBlock(1024, 1024, kernel_size=3, padding=1))  # 6
            layers[5].append(CNNBlock(1024, 1024, kernel_size=3, padding=1))
            self.model = nn.Sequential(*[nn.Sequential(*layer) for layer in layers])
        else:
            model, downsample, freeze_layers = pretrained_models.get(backbone_name)
            features = list(model.children())[:-2]
            for parameters in [feature.parameters() for i, feature in enumerate(features) if i < freeze_layers]:
                for parameter in parameters:
                    parameter.requires_grad = False
            self.model = nn.Sequential(*features, downsample)

    def forward(self, x):
        x = self.model(x)
        return x


class Yolov1(nn.Module):
    def __init__(self, backbone_name: str, grid_num=GRID_NUM, model_save_dir=MODEL_SAVE_DIR):
        super(Yolov1, self).__init__()
        self.model_save_dir = model_save_dir
        self.grid_num = grid_num
        self.backbone = Backbone(backbone_name)
        self.model_save_name = '{}_{}'.format(self.__class__.__name__, backbone_name)
        self.cls = nn.Sequential(
            nn.Linear(int(1024 * self.grid_num * self.grid_num), 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, int(self.grid_num * self.grid_num * 30)),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        x = x.view(-1, self.grid_num, self.grid_num, 30)
        #x_p, x_c = torch.split(x, [10, 20], dim=-1)
        #x_p = torch.sigmoid(x_p)
        #x_c = F.softmax(x_c, dim=-1)
        #x = torch.cat([x_p, x_c], dim=-1)
        x = torch.sigmoid(x)
        return x

    def save_model(self, step=None, optimizer=None, lr_scheduler=None, t_loss=None, v_loss=None, t_maps=None, v_maps=None, snap_save=False):
        save_name = self.model_save_name
        if snap_save:
            save_name += f'_{step}'
        self.save_safely(self.state_dict(), self.model_save_dir, save_name + '.pt')
        print('*** model weights saved successfully at {}!'.format(
            osp.join(self.model_save_dir, save_name + '.pt')))
        if optimizer and lr_scheduler and step is not None:
            temp = {
                'step': step,
                'train_loss_summary': t_loss,
                'valid_loss_summary': v_loss,
                'train_mean_aps': t_maps,
                'valid_mean_aps': v_maps,
                'lr_scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            self.save_safely(temp, self.model_save_dir, save_name + '_para.pt')
            print('*** auxiliary part saved successfully at {}!'.format(
                osp.join(self.model_save_dir, save_name + '.pt')))

    def load_model(self, optimizer=None, lr_scheduler=None, step=None):
        try:
            model_name = self.model_save_name
            if step is not None:
                model_name += f"_{step}"
            saved_model = torch.load(osp.join(self.model_save_dir, model_name + '.pt'),
                                     map_location='cpu')
            self.load_state_dict(saved_model)
            print('*** loading model weight successfully!')
        except Exception:
            print('*** loading model weight fail!')

        if optimizer and lr_scheduler is not None:
            try:
                temp = torch.load(osp.join(self.model_save_dir, self.model_save_name + '_para.pt'), map_location='cpu')
                lr_scheduler.load_state_dict(temp['lr_scheduler'])
                step = temp['step']
                train_loss_summary = temp['train_loss_summary']
                valid_loss_summary = temp['valid_loss_summary']
                train_mean_aps = temp['train_mean_aps']
                valid_mean_aps = temp['valid_mean_aps']
                print('*** loading optimizer&lr_scheduler&step successfully!')
                return step, train_loss_summary, valid_loss_summary, train_mean_aps, valid_mean_aps
            except Exception:
                print('*** loading optimizer&lr_scheduler&step fail!')
                return 0, [], [], [], []

    @staticmethod
    def save_safely(file, dir_path, file_name):
        r"""
        save the file safely, if detect the file name conflict,
        save the new file first and remove the old file
        """
        if not osp.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print('*** dir not exist, created one')
        save_path = osp.join(dir_path, file_name)
        if osp.exists(save_path):
            temp_name = save_path + '.temp'
            torch.save(file, temp_name)
            os.remove(save_path)
            os.rename(temp_name, save_path)
            print('*** found file conflict while saving, saved safely')
        else:
            torch.save(file, save_path)


if __name__ == '__main__':
    from torch import optim
    from lr_scheduler import WarmUpMultiStepLR

    x = torch.rand(2, 3, 448, 448)
    for name in ['yolo', 'resnet18', 'resnet50', 'vgg11', 'vgg16']:
        step = 0
        yolo_model = Yolov1(backbone_name=name)
        optimizer = optim.SGD(yolo_model.parameters(),
                              lr=LEARNING_RATE,
                              momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY)
        scheduler = WarmUpMultiStepLR(optimizer,
                                      milestones=STEP_LR_SIZES,
                                      gamma=STEP_LR_GAMMA,
                                      warm_up_factor=WARM_UP_FACTOR,
                                      warm_up_iters=WARM_UP_NUM_ITERS)
        yolo_model.save_model(optimizer=optimizer, lr_scheduler=scheduler, step=step)
        yolo_model.load_model(optimizer=optimizer, lr_scheduler=scheduler)
        print(yolo_model.model_save_name)
        y2 = yolo_model(x)
        print(f'y2.shape:{y2.shape}')
        del yolo_model
