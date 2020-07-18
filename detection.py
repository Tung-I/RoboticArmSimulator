import json
import os
import re
import fnmatch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import torch
import copy
from torch.autograd import Variable
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from model.roi_layers import nms

from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.net_utils import vis_detections

MODEL_PATH = '/home/tony/Object-Detection-Project/models/res50_coco/res50/coco/faster_rcnn_1_10_14657.pth' 
CLASS_NAME = ['car', 'chair', 'mug']

def get_model(load_path, agnostic=False, cuda=True):
    fasterRCNN = resnet(CLASS_NAME, 50, pretrained=False, class_agnostic=agnostic)
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_path))
    checkpoint = torch.load(load_path)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    fasterRCNN.eval()
    if cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
    return fasterRCNN

def prepare_variable():
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    support_ims = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    support_ims = support_ims.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    support_ims = Variable(support_ims)

    return im_data, im_info, num_boxes, gt_boxes, support_ims

def im_preprocess(im):
    im = im[:,:,::-1]  # rgb -> bgr
    target_size = cfg.TRAIN.SCALES[0]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_blob = im_list_to_blob([im])[0]
    info = np.array([[im_blob.shape[0], im_blob.shape[1], im_scale]], dtype=np.float32)

    return im_blob, info

def inference(np_im, model_path=MODEL_PATH):
    faster_rcnn = get_model(model_path)
    im_blob, info = im_preprocess(np_im)
    gt_boxes = torch.zeros((1, 1, 1))
    n_boxes = 0
    data = [im_blob, info, gt_boxes, n_boxes]

    with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

def vis_bbox(im, dets):
    """Visual debugging of detections."""
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        # class_name = CLASS_NAME[int(dets[i, 4]) - 1]
        class_name = ' '
        score = 0.99
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    return im

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)  # (max of H, max of W), but there is just one image actually
    num_images = len(ims)  # num_images = 1
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


# CLASS_NAME = ['airplane', 'car', 'chair', 'guitar', 'mug']
# cwd = os.getcwd()


# model_dir = cwd + '/models/simu_ft/' + 'res50' + "/" + 'simulated'
# fasterRCNN = get_model(model_dir)


# im_data, im_info, num_boxes, gt_boxes, support_ims = prepare_variable()


# n = 13
# nn = 2
# support_im_size = 320



# cwd = os.getcwd()
# np_dir = cwd + '/data/shots'
# im = np.load(np_dir + '/image_' + str(n) + '.npy')
# dets = np.load(np_dir + '/annotation_' + str(n) + '.npy')
# support_im = np.load(np_dir + '/closeup_' + str(n) + '_' + str(nn) + '.npy')

# # query and gt
# im2show = copy.deepcopy(im)
# im = im[:,:,::-1]  # rgb -> bgr
# target_size = cfg.TRAIN.SCALES[0]
# im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
#                 cfg.TRAIN.MAX_SIZE)
# im_blob = im_list_to_blob([im])[0]
# info = np.array([[im_blob.shape[0], im_blob.shape[1], im_scale]], dtype=np.float32)

# dets[:, :4] = dets[:, :4] * im_scale

# # support
# support_im = support_im[:,:,::-1] 
# support_im, support_im_scale = prep_im_for_blob(support_im, cfg.PIXEL_MEANS, support_im_size,
#                 cfg.TRAIN.MAX_SIZE)
# support_im_blob = im_list_to_blob([support_im])[0]

# im_blob = torch.from_numpy(im_blob).permute(2, 0, 1).contiguous().unsqueeze(0)
# dets = torch.from_numpy(dets).unsqueeze(0)
# support_im_blob = torch.from_numpy(support_im_blob).permute(2, 0, 1).contiguous().unsqueeze(0)
# info = torch.from_numpy(info)
# n_boxes = torch.from_numpy(np.array([dets.size(1)]))
# data = [im_blob, info, dets, n_boxes, support_im_blob]

# with torch.no_grad():
#     im_data.resize_(data[0].size()).copy_(data[0])
#     im_info.resize_(data[1].size()).copy_(data[1])
#     gt_boxes.resize_(data[2].size()).copy_(data[2])
#     num_boxes.resize_(data[3].size()).copy_(data[3])
#     support_ims.resize_(data[4].size()).copy_(data[4])

# # print(im_data[0].max())
# # print(im_data[0].min())
# # print(im_info[0])
# # raise Exception(' ')

# rois, cls_prob, bbox_pred, \
# rpn_loss_cls, rpn_loss_box, \
# RCNN_loss_cls, RCNN_loss_bbox, \
# rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, support_ims)

# scores = cls_prob.data
# boxes = rois.data[:, :, 1:5]

# if cfg.TEST.BBOX_REG:
#     # Apply bounding-box regression deltas
#     box_deltas = bbox_pred.data
#     if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#     # Optionally normalize targets by a precomputed mean and stdev
#         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#         box_deltas = box_deltas.view(1, -1, 4)

#     pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
#     pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
# else:
#     # Simply repeat the boxes, once for each class
#     pred_boxes = np.tile(boxes, (1, scores.shape[1]))

# # re-scale boxes to the origin img scale
# pred_boxes /= data[1][0][2].item()

# scores = scores.squeeze()
# pred_boxes = pred_boxes.squeeze()
# thresh = 0.05
# inds = torch.nonzero(scores[:,1]>thresh).view(-1)
# cls_scores = scores[:,1][inds]
# _, order = torch.sort(cls_scores, 0, True)
# cls_boxes = pred_boxes[inds, :]
# cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
# cls_dets = cls_dets[order]
# keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
# cls_dets = cls_dets[keep.view(-1).long()]

# im2show = vis_detections(im2show, ' ', cls_dets.cpu().numpy(), 0.4)

# # im_pred = vis_bbox(im, cls_dets.cpu().numpy())
# np.save(cwd + '/output/im_pred.npy', im2show)
# # print(cls_dets.size())
# print(cls_dets)