import argparse
import os

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from config import DEVICE
from model import Yolov1
from lr_scheduler import WarmUpMultiStepLR
from yolov1loss import Yolov1Loss
from augmentations import *
from dataset import *
from predict import tb_log_predict, calc_map


# from dataset import

def config_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    # 训练集与基础网络设定
    parser.add_argument('--voc_data_set_root', default='data/train/VOCdevkit',
                        help='data_set root directory path')
    parser.add_argument('--dataset', default='voc2007', choices=['voc2007', 'voc2012', 'coco2014'],
                        help='dataset to use in training')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    # 文件保存路径
    parser.add_argument('--save_dir', default='models/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir', default='logs/',
                        help='Directory for logging training')
    parser.add_argument('--name', default='',
                        help='Extra directory name for any variations')
    parser.add_argument('--save_step', default=1000, type=int,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_snap', default=2000, type=int,
                        help='Directory for saving checkpoint models')
    # 恢复训练
    parser.add_argument('--backbone', default='yolo', choices=['yolo', 'resnet18', 'resnet50', 'vgg11', 'vgg16'],
                        help='pre-trained base model name.')
    # 优化器参数设置
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--epochs', default=135, type=int,
                        help='Number of epochs to train model on')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--lr_mult_epochs', default=[75, 105], nargs='+',
                        help='Epochs to apply gamma to SGD LR')
    parser.add_argument('--warm_up_factor', default=0.1, type=float,
                        help='Warmup factor on LR for scheduler')
    parser.add_argument('--warm_up_epochs', default=1, type=int,
                        help="Warmup epochs on LR")
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda or not')

    args = parser.parse_args()
    return args


def train(args):
    if args.dataset == 'voc2007':
        train_set = [('2007', 'train')]
        val_set = [('2007', 'val')]
        train_transform = Yolov1Augmentation(dataset=args.dataset, size=448, percent_coord=True)
        valid_transform = Yolov1TestAugmentation(dataset=args.dataset, size=448, percent_coord=True)
        classes = VOC_CLASSES
    else:
        raise NotImplementedError(f'{args.dataset} dataset not implemented')

    # Get data
    print(f"Loading {args.dataset}...")
    train_data = VOCDetection(root=args.voc_data_set_root,
                              image_sets=train_set,
                              transform=train_transform,
                              dataset_name=args.dataset)
    valid_data = VOCDetection(root=args.voc_data_set_root,
                              image_sets=val_set,
                              transform=valid_transform,
                              dataset_name=args.dataset)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=detection_collate,
                              pin_memory=False)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=detection_collate,
                              pin_memory=False)
    steps_per_epoch = len(train_loader)
    print(f"There are {len(train_data)} training samples, forming {steps_per_epoch} {args.batch_size}-sized batches.")
    print(f"There are {len(valid_data)} validation samples, forming {len(valid_loader)} {args.batch_size}-sized batches.")

    # Save dir
    save_dir = os.path.join(args.save_dir, args.name)
    log_dir = os.path.join(args.log_dir, args.name)
    img_dir = os.path.join(args.save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    print(f"Logging to {log_dir}")

    # Setup logging
    train_logger = SummaryWriter(log_dir=os.path.join(log_dir, "train"), flush_secs=1)
    valid_logger = SummaryWriter(log_dir=os.path.join(log_dir, "valid"), flush_secs=1)

    # Setup training
    model = Yolov1(backbone_name=args.backbone, model_save_dir=save_dir)
    summary(model, input_size=(args.batch_size, 3, 448, 448))
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    milestones = [steps_per_epoch * ep for ep in args.lr_mult_epochs]
    warm_up_iters = args.warm_up_epochs * steps_per_epoch
    scheduler = WarmUpMultiStepLR(optimizer,
                                  milestones=milestones,
                                  gamma=args.gamma,
                                  warm_up_factor=args.warm_up_factor,
                                  warm_up_iters=warm_up_iters)
    step, train_loss_summary, valid_loss_summary, \
            train_mean_aps, valid_mean_aps = model.load_model(optimizer=optimizer, lr_scheduler=scheduler)
    criterion = Yolov1Loss()

    # Train model
    model.to(DEVICE)
    steps_to_finish = args.epochs * steps_per_epoch
    while step < steps_to_finish:
        model.train()
        train_losses = []
        for imgs, gt_boxes, gt_labels, gt_outs in tqdm(train_loader):
            imgs = imgs.to(DEVICE)
            gt_outs = gt_outs.to(DEVICE)
            model_outs = model(imgs)
            loss = criterion(model_outs, gt_outs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_logger.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step=step)
            scheduler.step()
            train_logger.add_scalar('loss', loss.item(), global_step=step)
            train_losses.append(loss.detach().cpu().item() * imgs.shape[0])
            if (step + 1) % args.save_step == 0:
                model.save_model(step, optimizer, scheduler, train_loss_summary, valid_loss_summary,
                        train_mean_aps, valid_mean_aps)
            if (step + 1) % args.save_snap == 0:
                model.save_model(step, optimizer, scheduler, train_loss_summary, valid_loss_summary,
                        train_mean_aps, valid_mean_aps, snap_save=True)
            step += 1

        model.eval()
        valid_losses = []
        for imgs, gt_boxes, gt_labels, gt_outs in tqdm(valid_loader):
            imgs = imgs.to(DEVICE)
            gt_outs = gt_outs.to(DEVICE)
            model_outs = model(imgs)
            loss = criterion(model_outs, gt_outs)
            valid_losses.append(loss.detach().cpu().item() * imgs.shape[0])

        avg_train_loss = sum(train_losses) / len(train_data)
        avg_valid_loss = sum(valid_losses) / len(valid_data)
        train_mean_ap, _ = calc_map(train_data, classes, model, valid_transform)
        valid_mean_ap, _ = calc_map(valid_data, classes, model, valid_transform)

        train_loss_summary.append(avg_train_loss)
        valid_loss_summary.append(avg_valid_loss)
        train_mean_aps.append(train_mean_ap)
        valid_mean_aps.append(valid_mean_ap)

        valid_logger.add_scalar('loss', avg_valid_loss, global_step=step)
        train_logger.add_scalar('mean ap', train_mean_ap, global_step=step)
        valid_logger.add_scalar('mean ap', valid_mean_ap, global_step=step)
        tb_log_predict('train', train_logger, step, train_data, model, 1199, valid_transform, img_dir)
        tb_log_predict('valid', valid_logger, step, valid_data, model, 380, valid_transform, img_dir)

        print(f"Step {step} | Train Loss: {avg_train_loss} | Valid Loss: {avg_valid_loss} | Train mAP: {train_mean_ap} | Valid mAP: {valid_mean_ap}")

    model.save_model(step, optimizer, scheduler, train_loss_summary, valid_loss_summary,
            train_mean_aps, valid_mean_aps, snap_save=True)


if __name__ == '__main__':
    args = config_parser()
    train(args)
