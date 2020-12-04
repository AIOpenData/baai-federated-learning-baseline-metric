from __future__ import division
import logging
import logging.handlers
import sys
import torch
import numpy as np
import json
import tqdm

logger = logging.getLogger(__name__)


def init_log(base_level=logging.INFO):
    """ initialize log output configuration """
    _formatter = logging.Formatter("%(asctime)s: %(filename)s: %(lineno)d: %(levelname)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(base_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_formatter)
    console_handler.setLevel(base_level)
    logger.addHandler(console_handler)


def load_json(result_json):
    """ load predicted and gold json file """
    with open(result_json, 'rb') as f:
        raw_data = json.load(f)
    return raw_data


def get_ann(image_id, annotations):
    """ get the gold annotation information """
    ann = []
    labels = []
    for j in range(len(annotations)):
        ann_item = annotations[j]
        if ann_item['image_id'] == image_id:
            cls_id = ann_item['category_id'] - 1
            x1 = ann_item['bbox'][0]  # xmin
            x2 = ann_item['bbox'][0] + ann_item['bbox'][2]  # xmax
            y1 = ann_item['bbox'][1]
            y2 = ann_item['bbox'][1] + ann_item['bbox'][3]
            labels.append(cls_id)
            if "score" in ann_item:
                pred_score = ann_item["score"]
                ann.append([x1, y1, x2, y2, pred_score, cls_id])
            else:
                ann.append([cls_id, x1, y1, x2, y2])
    return labels, np.array(ann)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """ returns the IoU of two bounding boxes """
    if not x1y1x2y2:
        # transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # get intersection area and union area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def helmet_image_false_detection(pred_no_wear_labels, pred_no_wear_boxes, target_no_wear_labels, target_no_wear_boxes,
                                 iou_threshold):
    """ check if the given helmet image is false detection """
    # if the given image predictions are all wear_helmet or empty, it is not false detection
    if len(pred_no_wear_labels) == 0:
        return 0, 0

    # if the given image gold labels are all wear_helmet or empty, it is false detection
    if len(target_no_wear_labels) == 0:
        return 1, 1

    # the predicted no_wear_helmet boxes number is more than twice as much as
    # the gold no_wear_helmet boxes number (No iou threshold requirements), it is false detection
    if len(pred_no_wear_labels) / len(target_no_wear_labels) > 2:
        # print("{} / {}".format(len(pred_no_wear_labels), len(target_no_wear_labels)))
        return 1, 1

    target_to_pred_no_wear_idx_list = []  # the indices of the boxes that have been detected

    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_no_wear_boxes, pred_no_wear_labels)):
        # if the detected boxes number is the same as gold boxes, break
        if len(target_to_pred_no_wear_idx_list) == len(target_no_wear_labels):
            break

        # find the max iou value and index of the iou of the predicted boxes and target boxes
        iou, box_idx = bbox_iou(pred_box.unsqueeze(0), target_no_wear_boxes).max(0)

        # the conf scores of the subsequent predicted boxes are smaller, they are useless
        # even if the ious are bigger
        # if correctly predict a no_wear_helmet box
        if (iou >= iou_threshold) and (box_idx not in target_to_pred_no_wear_idx_list):
            target_to_pred_no_wear_idx_list.append(box_idx)
            return 0, 1
    return 1, 1


def helmet_image_missed_detection(pred_no_wear_labels, pred_no_wear_boxes, target_no_wear_labels, target_no_wear_boxes,
                                 iou_threshold):
    """ check if the given helmet image is missed detection """
    # if the given image gold labels are all wear_helmet or empty, it is not missed detection
    if len(target_no_wear_labels) == 0:
        return 0, 0

    # if the given image predictions are all wear_helmet or empty, it is missed detection
    if len(pred_no_wear_labels) == 0:
        return 1, 1

    target_to_pred_no_wear_idx_list = []  # the indices of the boxes that have been detected

    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_no_wear_boxes, pred_no_wear_labels)):
        # if the detected boxes number is the same as gold boxes, break
        if len(target_to_pred_no_wear_idx_list) == len(target_no_wear_labels):
            break

        # find the max iou value and index of the iou of the predicted boxes and target boxes
        iou, box_idx = bbox_iou(pred_box.unsqueeze(0), target_no_wear_boxes).max(0)

        # the conf scores of the subsequent predicted boxes are smaller, they are useless
        # even if the ious are bigger
        # if correctly predict a no_wear_helmet box
        if (iou >= iou_threshold) and (box_idx not in target_to_pred_no_wear_idx_list):
            target_to_pred_no_wear_idx_list.append(box_idx)
            return 0, 1
    return 1, 1


def helmet_image_object_detection(pred_no_wear_labels, pred_no_wear_boxes, target_no_wear_labels, target_no_wear_boxes,
                                 iou_threshold):
    """ detect the predicted no_wear_helmet objects for the given helmet image """
    # if the given image is a background image
    if len(target_no_wear_labels) == 0:
        return 0, 0

    correct_number, total_number = 0, len(target_no_wear_labels)

    # if the given image predictions are all wear_helmet or empty
    if len(pred_no_wear_labels) == 0:
        return correct_number, total_number

    target_to_pred_no_wear_idx_list = []  # the indices of the boxes that have been detected

    for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_no_wear_boxes, pred_no_wear_labels)):
        # if the detected boxes number is the same as gold boxes, break
        if len(target_to_pred_no_wear_idx_list) == len(target_no_wear_labels):
            break

        # find the max iou value and index of the iou of the predicted boxes and target boxes
        iou, box_idx = bbox_iou(pred_box.unsqueeze(0), target_no_wear_boxes).max(0)

        # the conf scores of the subsequent predicted boxes are smaller, they are useless
        # even if the ious are bigger
        # if correctly predict a no_wear_helmet box
        if (iou >= iou_threshold) and (box_idx not in target_to_pred_no_wear_idx_list):
            target_to_pred_no_wear_idx_list.append(box_idx)
            correct_number += 1

    return correct_number, total_number
