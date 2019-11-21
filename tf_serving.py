import numpy as np
import sys
import tensorflow as tf
import cv2
sys.path.append("./models/")
sys.path.append("./models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util
from definitions import ROOT_DIR
from PIL import Image
import io
import base64
import os
import json
import requests


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
        
        Arguments:
        box1 -- first box, numpy array with coordinates (x1, y1, x2, y2)
        box2 -- second box, numpy array with coordinates (x1, y1, x2, y2)
        """

    xi1 = max(box1[0], box2[0])
    yi1 = max(box2[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou


def filter_boxes(scores, threshold = 0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

        """
    index = np.where(scores > threshold)
    return index


def non_max_suppression(boxes, scores, classes, iou_threshold=0.5):
    """
        Applies Non-max suppression (NMS) to set of boxes.

        """
    box_with_high_iou = boxes[0]
    # Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold.
    for box in boxes: # descending order of scores
        box_selected = box
        for index, box in enumerate(boxes):
            iou_score = iou(box_selected, box)
            if iou_score > iou_threshold:
                # remove box
                exclude_boxes.append(index)

    return

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = ROOT_DIR + '/workspace/training_demo/exported_graphs/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = ROOT_DIR + '/workspace/training_demo/data/label_map.pbtxt'

# Number of classes to detect
NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



with tf.gfile.GFile(('test_image/hand1.jpg'), 'rb') as fid:
    encoded_input_string = base64.b64encode(fid.read())
input = cv2.imread('test_image/hand1.jpg')
input_string = encoded_input_string.decode("utf-8")
instance = [{'image_bytes': {"b64": input_string}}]
data = json.dumps({"instances": instance})
url = "http://localhost:8501/v1/models/faster_rcnn:predict"
json_response = requests.post(url=url, data=data)
response = json.loads(json_response.text)
boxes = response['predictions'][0]['detection_boxes']
classes = response['predictions'][0]['detection_classes']
scores = response['predictions'][0]['detection_scores']
scores = np.squeeze(scores)
boxes = np.squeeze(boxes)
classes = np.squeeze(classes).astype(np.int32)

print (scores)
exit(1)
index = filter_boxes(scores)
boxes = boxes[index]
scores = scores[index]
classes = classes[index]
print (boxes)
print (scores)

#boxes, scores, classes = non_max_suppression(boxes, scores, classes)
#print (boxes)
#print (scores)
#print (classes)
#exit(1)
vis_util.visualize_boxes_and_labels_on_image_array(
            input,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
cv2.imshow('object detection', cv2.resize(input, (800, 600)))
cv2.waitKey()