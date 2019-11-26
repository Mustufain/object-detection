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
import time

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


def prepare_request(image):
    with tf.gfile.GFile(image, 'rb') as fid:
        encoded_input_string = base64.b64encode(fid.read())
    input = cv2.imread(image)
    input_string = encoded_input_string.decode("utf-8")
    instance = [{'image_bytes': {"b64": input_string}}]
    data = json.dumps({"instances": instance})
    return [input, data]


def get_iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
        
        Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        """
    # ymin, xmin, ymax, xmax = box
    print (box1, box2)
    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou


def non_max_suppression(boxes, probs, overlapthresh=0.5):
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


from sagemaker.tensorflow.serving import Model,  Predictor
from sagemaker.tensorflow import TensorFlowPredictor
#model = Model(model_data='s3://my-sage-maker-test/faster_rcnn.tar.gz',
#              role='arn:aws:iam::646335046581:role/service-role/AmazonSageMaker-ExecutionRole-20191125T144296',
#              framework_version="1.12")
#predictor = model.deploy(initial_instance_count=1,
#                         instance_type='ml.p3.16xlarge')

#exit(1)
image = 'test_image/test_4.jpg'
endpoint = 'sagemaker-tensorflow-serving-2019-11-25-18-54-08-150'
#input, data = prepare_request(image)
with tf.gfile.GFile('test_image/test_3.jpg', 'rb') as fid:
    encoded_jpg = base64.b64encode(fid.read())
image_base64 = encoded_jpg.decode()
data = {"instances": [{"image_bytes": {"b64": image_base64}}]}
predictor = Predictor(endpoint)
response = predictor.predict(data)
print (response)



exit(1)
start_time = time.time()
image = 'test_image/test_4.jpg'
input, data = prepare_request(image)
boxes, classes, scores = get_predictions(data)
boxes, classes, scores = post_process(boxes, classes, scores)
#visualize(input, boxes, classes, scores)
#cv2.imshow('window', cv2.resize(input, (800, 600)))
#cv2.waitKey()
end_time = time.time()

execution_time = end_time - start_time
print (execution_time)



# on my machine inference time = 26 seconds
# train = 1.12 tf version