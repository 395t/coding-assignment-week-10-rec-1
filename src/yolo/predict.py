import torch
from config import GRID_NUM, DEVICE
from dataset import VOC_CLASSES
from model import Yolov1
from matplotlib import pyplot as plt
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
from os import path as osp
import os
from numpy.random import shuffle
from tqdm import tqdm


def draw_box(img_np, boxes_np, tags_np, scores_np=None, relative_coord=False, save_path=None, img_id=None, step=None):
    if scores_np is None:
        scores_np = [1.0 for i in tags_np]
    # img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_np.shape
    if relative_coord and len(boxes_np) > 0:
        boxes_np = np.array([
            boxes_np[:, 0] * w,
            boxes_np[:, 1] * h,
            boxes_np[:, 2] * w,
            boxes_np[:, 3] * h,
        ]).T
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    for box, tag, score in zip(boxes_np, tags_np, scores_np):
        from dataset import VOC_CLASSES as LABELS
        tag = int(tag)
        label_name = LABELS[tag]
        display_txt = '%s: %.2f' % (label_name, score)
        coords = (box[0], box[1]), box[2] - box[0] + 1, box[3] - box[1] + 1
        color = colors[tag]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        ax.text(box[0], box[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    if img_id and step:
        ax.set_title(f"BBoxes of IMG {img_id} at Step {step}")
    ax.imshow(img_np)
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    return fig, ax


def decoder(pred, obj_thres=0.1):
    r"""
    :param pred: the output of the yolov1 model, should be tensor of [1, grid_num, grid_num, 30]
    :param obj_thres: the threshold of objectness
    :return: list of [c, [boxes, labels]], boxes is [:4], labels is [4]
    """
    pred = pred.cpu()
    assert pred.shape[0] == 1
    # i for W, j for H
    res = [[] for i in range(len(VOC_CLASSES))]
    # print(res)
    for h in range(GRID_NUM):
        for w in range(GRID_NUM):
            better_box = pred[0, h, w, :5] if pred[0, h, w, 4] > pred[0, h, w, 9] else pred[0, h, w, 5:10]
            if better_box[4] < obj_thres:
                continue
            better_box_xyxy = torch.FloatTensor(better_box.size())
            # print(f'grid(cx,cy), (w,h), obj:{better_box}')
            better_box_xyxy[:2] = better_box[:2] / float(GRID_NUM) - 0.5 * better_box[2:4]
            better_box_xyxy[2:4] = better_box[:2] / float(GRID_NUM) + 0.5 * better_box[2:4]
            better_box_xyxy[0:4:2] += (w / float(GRID_NUM))
            better_box_xyxy[1:4:2] += (h / float(GRID_NUM))
            better_box_xyxy = better_box_xyxy.clamp(max=1.0, min=0.0)
            score, cls = pred[0, h, w, 10:].max(dim=0)
            # print(f'pre_cls_shape:{pred[0, w, h, 10:].shape}')
            from dataset import VOC_CLASSES as LABELS
            # print(f'score:{score}\tcls:{cls}\ttag:{LABELS[cls]}')
            better_box_xyxy[4] = score * better_box[4]
            res[cls].append(better_box_xyxy)
    # print(res)
    for i in range(len(VOC_CLASSES)):
        if len(res[i]) > 0:
            # res[i] = [box.unsqueeze(0) for box in res[i]]
            res[i] = torch.stack(res[i], 0)
        else:
            res[i] = torch.tensor([])
    # print(res)
    return res


def _nms(boxes, scores, overlap=0.5, top_k=None):
    r"""
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # boxes = boxes.detach()
    # keep shape [num_prior] type: Long
    keep = scores.new(scores.size(0)).zero_().long()
    # print('keep.shape:{}'.format(keep.shape))
    # tensor.numel()用于计算tensor里面包含元素的总数，i.e. shape[0]*shape[1]...
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # print('x1:{}\ny1:{}\nx2:{}\ny2:{}'.format(x1, y1, x2, y2))
    # area shape[prior_num], 代表每个prior框的面积
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # print(f'idx:{idx}')
    # I = I[v >= 0.01]
    if top_k is not None:
        # indices of the top-k largest vals
        idx = idx[-top_k:]
    # keep = torch.Tensor()
    count = 0
    # Returns the total number of elements in the input tensor.
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # torch.index_select(input, dim, index, out=None)
        # 将input里面dim维度上序号为idx的元素放到out里面去
        # >>> x
        # tensor([[1, 2, 3],
        #         [3, 4, 5]])
        # >>> z=torch.index_select(x,0,torch.tensor([1,0]))
        # >>> z
        # tensor([[3, 4, 5],
        #         [1, 2, 3]])
        xx1 = x1[idx]
        # torch.index_select(x1, 0, idx, out=xx1)
        yy1 = y1[idx]
        # torch.index_select(y1, 0, idx, out=yy1)
        xx2 = x2[idx]
        # torch.index_select(x2, 0, idx, out=xx2)
        yy2 = y2[idx]
        # torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        # 将除置信度最高的prior框外的所有框进行clip以计算inter大小
        # print(f'xx1.shape:{xx1.shape}')
        xx1 = torch.clamp(xx1, min=float(x1[i]))
        yy1 = torch.clamp(yy1, min=float(y1[i]))
        xx2 = torch.clamp(xx2, max=float(x2[i]))
        yy2 = torch.clamp(yy2, max=float(y2[i]))
        # w.resize_as_(xx2)
        # h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        # torch.le===>less and equal to
        idx = idx[IoU.le(overlap)]
    # print(keep, count)
    # keep 包含置信度从大到小的prior框的indices，count表示数量
    # print('keep.shape:{},count:{}'.format(keep.shape, count))
    return keep, count


def img_to_tensor_batch(img_path, size=(448, 448)):
    img = Image.open(img_path)
    img_resize = img.resize(size, PIL.Image.BILINEAR)
    img_tensor = transforms.ToTensor()(img_resize).unsqueeze(0)
    # print(f'img_tensor:{img_tensor.shape}')
    # print(f'img_tensor:{img_tensor}')
    return img_tensor, img


def predict_one_img(img_path, model):
    # model = Yolov1(backbone_name=backbone_name)
    # model.load_model()
    img_tensor, img = img_to_tensor_batch(img_path)
    boxes, tags, scores = predict(img_tensor, model)
    img = np.array(img)
    fig, ax = draw_box(img_np=img, boxes_np=boxes, scores_np=scores, tags_np=tags, relative_coord=True)
    plt.close(fig)


def tb_log_predict(data_type, logger, step, dataset, model, img_idx, transform, save_dir):
    model.eval()
    img = dataset.pull_image(img_idx)
    img_id, gt = dataset.pull_anno(img_idx)
    gt = np.array(gt)
    boxes_gt = gt[:, :4]
    tags_gt = gt[:, 4]
    img_tensor, _, _ = transform(img, boxes_gt, tags_gt)
    img_tensor = img_tensor.unsqueeze(0)

    boxes_p, tags_p, scores_p = predict(img_tensor, model)

    fig_p, ax_p = draw_box(img_np=img, boxes_np=boxes_p, tags_np=tags_p, scores_np=scores_p, relative_coord=True,
                           save_path=os.path.join(save_dir, f"{data_type}_{step}.jpg"), img_id=img_id, step=step)
    logger.add_figure('pred visualization', fig_p, global_step=step)
    fig_gt, ax_gt = draw_box(img_np=img, boxes_np=boxes_gt, tags_np=tags_gt, relative_coord=False,
                             save_path=os.path.join(save_dir, f"{data_type}_gt.jpg"), img_id=img_id, step=step)
    logger.add_figure('gold visualization', fig_gt, global_step=step)


def predict(img_tensor, model):
    model.eval()
    img_tensor, model = img_tensor.to(DEVICE), model.to(DEVICE)
    with torch.no_grad():
        out = model(img_tensor)
        # out:list[tensor[, 5]]
        out = decoder(out, obj_thres=0.3)
        boxes, tags, scores = [], [], []
        for cls, pred_target in enumerate(out):
            if pred_target.shape[0] > 0:
                # print(pred_target.shape)
                b = pred_target[:, :4]
                p = pred_target[:, 4]
                # print(b, p)
                keep_idx, count = _nms(b, p, overlap=0.5)
                # keep:[, 5]
                keep = pred_target[keep_idx]
                for box in keep[..., :4]: boxes.append(box)
                for tag in range(count): tags.append(torch.LongTensor([cls]))
                for score in keep[..., 4]: scores.append(score)
        # print(f'*** boxes:{boxes}\ntags:{tags}\nscores:{scores}')
        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0).numpy()  # .squeeze(dim=0)
            tags = torch.stack(tags, 0).numpy()  # .squeeze(dim=0)
            scores = torch.stack(scores, 0).numpy()  # .squeeze(dim=0)
            # print(f'*** boxes:{boxes}\ntags:{tags}\nscores:{scores}')
        else:
            boxes = torch.FloatTensor([]).numpy()
            tags = torch.LongTensor([]).numpy()  # .squeeze(dim=0)
            scores = torch.FloatTensor([]).numpy()  # .squeeze(dim=0)
        # img, boxes, tags, scores = np.array(img), np.array(boxes), np.array(tags), np.array(scores)
        return boxes, tags, scores


def calc_map(dataset, classes, model, valid_transform, iou_threshold=0.5):
    # Idea from https://github.com/Cartucho/mAP/blob/master/main.py
    gt_counter, p_counter = {}, {}
    for idx in tqdm(range(len(dataset))):
        img = dataset.pull_image(idx)
        _, gt = dataset.pull_anno(idx)
        gt = np.array(gt)
        img_tensor, _, _ = valid_transform(img, gt[:, :4], gt[:, 4])
        img_tensor = img_tensor.unsqueeze(0)

        # Count ground truths
        taken = []
        for bb in gt:
            label = bb[4]
            if not label in gt_counter:
                gt_counter[label] = 0
            gt_counter[label] += 1
            taken.append(False)

        # Predictions
        boxes, tags, scores = predict(img_tensor, model)
        h, w, _ = img.shape
        if len(boxes) > 0:
            boxes = np.array([
                boxes[:, 0] * w,
                boxes[:, 1] * h,
                boxes[:, 2] * w,
                boxes[:, 3] * h,
            ]).T
        tags = [t[0] for t in tags]
        preds = list(zip(scores, tags, boxes))
        preds = sorted(preds, key=lambda x: x[0], reverse=True)

        # Match
        taken = [False for _ in range(len(gt))]
        for score, tag, box in preds:
            best_iou = 0
            best_idx = 0
            for idx in range(len(gt)):
                if tag == gt[idx][4]:
                    # Compute IOU
                    bb = gt[idx][:4]
                    bi = [max(box[0], bb[0]), max(box[1], bb[1]), min(box[2], bb[2]), min(box[3], bb[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        union = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (box[2] - box[0] + 1) \
                                * (box[3] - box[1] + 1) - iw * ih
                        iou = iw * ih / union
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx

            # Match if ground truth not taken
            if not taken[best_idx] and best_iou >= iou_threshold:
                taken[best_idx] = True
                if not tag in p_counter:
                    p_counter[tag] = []
                p_counter[tag].append((score, True))
            else:
                if not tag in p_counter:
                    p_counter[tag] = []
                p_counter[tag].append((score, False))

    # Sort predictions by confidence
    for key in p_counter.keys():
        p_counter[key] = sorted(p_counter[key], reverse=True)

    # Formulate precision and recall
    prec_counter, rec_counter, ap_dict = {}, {}, {}
    for key in p_counter.keys():
        prec_counter[key] = [0.0]
        rec_counter[key] = [0.0]
        tp, fp = 0, 0
        for confi, has_match in p_counter[key]:
            if has_match:
                tp += 1
            else:
                fp += 1
            prec_counter[key].append(tp / (tp + fp))
            if key not in gt_counter:
                rec_counter[key].append(0.0)
            else:
                rec_counter[key].append(tp / gt_counter[key])
        prec_counter[key].append(0.0)
        rec_counter[key].append(1.0)

        ap = 0.0
        best_prec = 0.0
        for i in range(len(prec_counter[key])-1, 0, -1):
            if prec_counter[key][i] > best_prec:
                best_prec = prec_counter[key][i]
            ap += best_prec * (rec_counter[key][i] - rec_counter[key][i-1])
        ap_dict[key] = {
            'AP': ap,
            'TP': tp,
            'FP': fp,
            'GT': 0 if key not in gt_counter else gt_counter[key],
        }

    # Fill missing values
    for key in range(len(classes)):
        if key not in ap_dict:
            ap_dict[key] = {'AP': 0.0, 'TP': 0, 'FP': 0, 'GT': 0 if key not in gt_counter else gt_counter[key]}

    # Calculate mean AP
    aps = [ap_dict[key]['AP'] for key in range(len(classes))]
    mean_ap = sum(aps) / len(aps)

    return mean_ap, ap_dict


"""
Below are from https://github.com/abeardear/pytorch-YOLO-v1/blob/master/eval_voc.py
"""
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.
    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(dataset, model, transform, classes=VOC_CLASSES, threshold=0.5, use_07_metric=False):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    preds, target = {}, {}
    for c in classes:
        preds[c] = []
    for idx in tqdm(range(len(dataset))):
        img = dataset.pull_image(idx)
        img_id, gt = dataset.pull_anno(idx)
        gt = np.array(gt)
        img_tensor, _, _ = transform(img, gt[:, :4], gt[:, 4])
        img_tensor = img_tensor.unsqueeze(0)

        for bb in gt:
            if (img_id, bb[4]) not in target:
                target[(img_id, classes[int(bb[4])])] = []
            target[(img_id, classes[int(bb[4])])].append(bb[:4])

        boxes, tags, scores = predict(img_tensor, model)
        h, w, _ = img.shape
        if len(boxes) > 0:
            boxes = np.array([
                boxes[:, 0] * w,
                boxes[:, 1] * h,
                boxes[:, 2] * w,
                boxes[:, 3] * h,
            ]).T
        tags = [t[0] for t in tags]
        ps = list(zip(scores, tags, boxes))
        for score, tag, box in ps:
            preds[classes[int(tag)]].append([img_id, score, *box])

    print(preds)
    print(target)

    aps = []
    for i,class_ in enumerate(classes):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = 0
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            continue
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)

                    overlaps = inters/union
                    if overlaps >= threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    mean_ap = np.mean(aps)
    print('---map {}---'.format(mean_ap))
    return mean_ap
######################################################################################


if __name__ == '__main__':
    # test:
    # fake_pred = torch.rand(1, GRID_NUM, GRID_NUM, 30)
    # decoder(fake_pred)
    CONTINUE = False  # continue from breakpoint
    model = Yolov1(backbone_name='resnet50')
    model.load_model()
    # predict_one_img('../test_img/000001.jpg', model)
    # test_img_dir = '../test_img'
    test_img_dir = '/Users/chenlinwei/Dataset/VOC0712/VOC2012test/JPEGImages'
    for root, dirs, files in os.walk(test_img_dir, topdown=True):
        if test_img_dir == root:
            print(root, dirs, files)
            files = [i for i in files if any([j in i for j in ['jpg', 'png', 'jpeg', 'gif', 'tiff']])]
            shuffle(files)
            if CONTINUE:
                with open(osp.join(test_img_dir, 'tested.txt'), 'a') as _:
                    pass
                with open(osp.join(test_img_dir, 'tested.txt'), 'r') as txt:
                    txt = txt.readlines()
                    txt = [i.strip() for i in txt]
                    print(txt)
                    files = [i for i in files if i not in txt]
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f'*** testing:{file_path}')
                    predict_one_img(file_path, model)
                    with open(osp.join(test_img_dir, 'tested.txt'), 'a') as txt:
                        txt.write(file + '\n')
            else:
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f'*** testing:{file_path}')
                    predict_one_img(file_path, model)
