import numpy as np
import sys
import tensorflow as tf
import cv2
sys.path.append("./models/")
sys.path.append("./models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util
from definitions import ROOT_DIR
import base64
import json
import requests
import time

def prepare_request(image):
    with tf.gfile.GFile(image, 'rb') as fid:
        encoded_input_string = base64.b64encode(fid.read())
    input = cv2.imread(image)
    input_string = encoded_input_string.decode("utf-8")
    instance = [{'image_bytes': {"b64": input_string}}]
    data = json.dumps({"instances": instance})
    with open('input.json', 'w') as f:
        f.write(data)
    return [input, data]


def non_max_suppression(boxes, probs, overlapthresh=0.9):
    """
        Applies Non-max suppression (NMS) to set of boxes.

        """

    if len(boxes) == 0:
        return []
        # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapthresh)[0])))

    # return only the bounding boxes that were picked
    return (pick, boxes[pick])


def get_predictions(data):

    url = "http://localhost:8501/v1/models/faster_rcnn:predict"
    json_response = requests.post(url=url, data=data)
    response = json.loads(json_response.text)
    boxes = response['predictions'][0]['detection_boxes']
    classes = response['predictions'][0]['detection_classes']
    scores = response['predictions'][0]['detection_scores']
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    return [boxes, classes, scores]


def post_process(boxes, classes, scores):
    index = np.nonzero(scores)
    scores = scores[index]
    boxes = boxes[index]
    classes = classes[index]
    id, value = non_max_suppression(boxes, scores)
    boxes = boxes[id]
    classes = classes[id]
    scores = scores[id]
    return [boxes, classes, scores]


def visualize(input, boxes, classes, scores):
    vis_util.visualize_boxes_and_labels_on_image_array(
        input,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)


if __name__ == '__main__':

    image = sys.argv[1]
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = ROOT_DIR + '/workspace/training_demo/data/label_map.pbtxt'

    # Number of classes to detect
    NUM_CLASSES = 1

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `1`, we know that this corresponds
    # to `hand`.  Here we use internal utility functions,
    # but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    start_time = time.time()
    input, data = prepare_request(image)
    boxes, classes, scores = get_predictions(data)
    boxes, classes, scores = post_process(boxes, classes, scores)
    visualize(input, boxes, classes, scores)
    cv2.imshow('window', cv2.resize(input, (800, 600)))
    keypress = cv2.waitKey()
