import logging.handlers
import sys
import torch
import traceback

from terminaltables import AsciiTable
from utils.options import args_parser
from utils.utils import get_ann, helmet_image_false_detection, helmet_image_missed_detection, \
    helmet_image_object_detection, init_log, load_json

args = args_parser()

logger = logging.getLogger(__name__)


def evaluate_helmet_image_sgcc_score(predicted_file_json_path: str, gold_json_file_path: str, iou_threshold: float,
                                     false_detection_weight: float,
                                     missed_detection_weight: float, object_detection_weight: float):
    """ calculate the sgcc helmet image score by the predicted and gold json file """
    try:
        gt_data = load_json(gold_json_file_path)
        pred_data = load_json(predicted_file_json_path)

        # load the names of categories
        class_name_list = []
        for class_item in gt_data['categories']:
            if isinstance(class_item['name'], list):
                class_name_list.append(class_item['name'][0])
            else:
                class_name_list.append(class_item['name'])

        class_name_dict = {'wear_helmet_label': 0, 'no_helmet_label': 1}

        # traverse the images, a batch of one picture
        false_detection_count = 0
        detection_no_wear_total_count = 0
        missed_detection_count = 0
        gold_no_wear_total_count = 0
        object_detection_correct_count = 0
        object_detection_total_count = 0
        for i in range(len(gt_data['images'])):
            image_id = gt_data['images'][i]['id']
            # load gold annotations，ann_gt = n * [cls_id, x1, y1, x2, y2]
            labels_gt, ann_gt = get_ann(image_id, gt_data['annotations'])
            # load predicted annotations，ann_pred = n * [x1, y1, x2, y2, pred_score, cls_id]
            _, ann_pred = get_ann(image_id, pred_data)
            # sort the ann pred list by the confidence pred scores in a descending order
            if len(ann_pred):
                ann_pred = ann_pred[(-ann_pred[:, 4]).argsort()]

            ann_pred = torch.Tensor(ann_pred)
            ann_gt = torch.Tensor(ann_gt)

            # predicted no_wear boxes and labels
            if len(ann_pred) == 0:
                pred_no_wear_indices, pred_no_wear_labels, pred_no_wear_boxes = [], [], []
            else:
                pred_no_wear_indices = torch.where(ann_pred[:, -1] == class_name_dict['no_helmet_label'])
                pred_no_wear_labels = ann_pred[:, -1][pred_no_wear_indices]
                pred_no_wear_boxes = ann_pred[:, :4][pred_no_wear_indices]

            # target no_wear boxes and labels
            if len(ann_gt) == 0:
                target_no_wear_indices, target_no_wear_labels, target_no_wear_boxes = [], [], []
            else:
                target_no_wear_indices = torch.where(ann_gt[:, 0] == class_name_dict['no_helmet_label'])
                target_no_wear_labels = ann_gt[:, 0][target_no_wear_indices]
                target_no_wear_boxes = ann_gt[:, 1:][target_no_wear_indices]

            false_detection_number, detection_no_wear_number = helmet_image_false_detection(
                pred_no_wear_labels=pred_no_wear_labels, pred_no_wear_boxes=pred_no_wear_boxes,
                target_no_wear_labels=target_no_wear_labels, target_no_wear_boxes=target_no_wear_boxes,
                iou_threshold=iou_threshold)
            false_detection_count += false_detection_number
            detection_no_wear_total_count += detection_no_wear_number

            missed_detection_number, gold_no_wear_number = helmet_image_missed_detection(
                pred_no_wear_labels=pred_no_wear_labels, pred_no_wear_boxes=pred_no_wear_boxes,
                target_no_wear_labels=target_no_wear_labels, target_no_wear_boxes=target_no_wear_boxes,
                iou_threshold=iou_threshold)
            missed_detection_count += missed_detection_number
            gold_no_wear_total_count += gold_no_wear_number

            object_detection_correct_number, object_detection_total_number = helmet_image_object_detection(
                pred_no_wear_labels=pred_no_wear_labels, pred_no_wear_boxes=pred_no_wear_boxes,
                target_no_wear_labels=target_no_wear_labels, target_no_wear_boxes=target_no_wear_boxes,
                iou_threshold=iou_threshold)
            object_detection_correct_count += object_detection_correct_number
            object_detection_total_count += object_detection_total_number

        false_detection_rate = (false_detection_count / detection_no_wear_total_count) if (
                detection_no_wear_total_count != 0) else 0
        missed_detection_rate = (missed_detection_count / gold_no_wear_total_count) if (
                gold_no_wear_total_count != 0) else 0
        object_detection_correct_rate = (object_detection_correct_count / object_detection_total_count) if (
                object_detection_total_count != 0) else 0

        logger.info("false_detection_rate: {} / {} = {}".format(false_detection_count, detection_no_wear_total_count,
                                                                false_detection_rate))
        logger.info("missed_detection_rate: {} / {} = {}".format(missed_detection_count, gold_no_wear_total_count,
                                                                 missed_detection_rate))
        logger.info("object_detection_correct_rate: {} / {} = {}".format(object_detection_correct_count,
                                                                         object_detection_total_count,
                                                                         object_detection_correct_rate))

        sgcc_helmet_image_score = 1 - (
                false_detection_weight * false_detection_rate + missed_detection_weight * missed_detection_rate + object_detection_weight * (
                1 - object_detection_correct_rate))

        logger.info("evaluation for {} and {}\n".format(predicted_file_json_path, gold_json_file_path))
        ap_table = [["false detection rate", "missed detection rate", "object detection correct rate",
                     "sgcc helmet image score"]]
        ap_table += [
            [false_detection_rate, missed_detection_rate, object_detection_correct_rate, sgcc_helmet_image_score]]
        logger.info("\n{}\n".format(AsciiTable(ap_table).table))

        return float('{:.8f}'.format(sgcc_helmet_image_score)), "评测成功"
    except Exception as e:
        return -1, "格式错误"
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        logger.info('an error occurred on line {} in statement {}'.format(line, text))

        return -1, "格式错误"


def entrance(predicted_file_json_path: str, gold_json_file_path: str):
    score, message = evaluate_helmet_image_sgcc_score(predicted_file_json_path=predicted_file_json_path,
                                                      gold_json_file_path=gold_json_file_path, iou_threshold=0.5,
                                                      false_detection_weight=0.3,
                                                      missed_detection_weight=0.5, object_detection_weight=0.2)
    if message != "评测成功":
        status = 0
    else:
        status = 1

    return score, message, status


if __name__ == "__main__":
    # initialize log output configuration
    init_log(logging.INFO)

    # set predicted and gold json file paths
    helmet_predicted_json_path = args.contestant_submitted_file_name
    helmet_gold_json_path = "test.json"

    sgcc_helmet_image_score = entrance(predicted_file_json_path=helmet_predicted_json_path,
                                       gold_json_file_path=helmet_gold_json_path)

    logger.info("sgcc helmet image score: {}".format(sgcc_helmet_image_score))
