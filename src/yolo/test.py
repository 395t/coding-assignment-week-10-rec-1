import argparse

from pprint import pprint

from dataset import VOCDetection, VOC_CLASSES
from model import Yolov1
from predict import calc_map
from augmentations import Yolov1TestAugmentation
from lr_scheduler import WarmUpMultiStepLR
from torch.optim import SGD


def voc_eval(args):
    set_dir = {
        'train': 'train',
        'val': 'train',
        'test': 'test'
    }
    valid_transform = Yolov1TestAugmentation(dataset=f'voc{args.year}', size=448, percent_coord=True)
    dataset = VOCDetection(root=f'data/{set_dir[args.set] if args.year != 2012 else "train"}/VOCdevkit',
                           image_sets=[(f'{args.year}', args.set)],
                           transform=valid_transform,
                           dataset_name=f'voc{args.year}')
    model = Yolov1(backbone_name=args.backbone, model_save_dir=f'models/voc{args.year}_{args.backbone}')
    optimizer = SGD(model.parameters(),
                    lr=1e-3,
                    momentum=0.9,
                    weight_decay=5e-4)
    scheduler = WarmUpMultiStepLR(optimizer,
                                  milestones=[1, 1],
                                  gamma=0.1,
                                  warm_up_factor=0.1,
                                  warm_up_iters=1)
    _ = model.load_model(optimizer=optimizer, lr_scheduler=scheduler, step=args.step)

    model.eval()
    mean_ap, ap_dict = calc_map(dataset, VOC_CLASSES, model, valid_transform)

    print("mAP:", mean_ap)
    pprint(ap_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch Evaluation')
    parser.add_argument('--set', default='test', choices=['test', 'train', 'valid'], help='dataset to evaluate on')
    parser.add_argument('--year', default=2007, type=int, help='dataset year')
    parser.add_argument('--backbone', default='yolo', choices=['yolo', 'resnet18', 'resnet50', 'vgg11', 'vgg16'])
    parser.add_argument('--step', default=None)

    args = parser.parse_args()
    voc_eval(args)
