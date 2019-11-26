import tensorflow as tf
from scipy.io import loadmat
import os
import sys
from tqdm import tqdm
import io
from PIL import Image
sys.path.append("../../models/research/object_detection")
sys.path.append("../../")
from utils import dataset_util
from definitions import  TRAIN_IMAGES, TRAIN_ANNOT,VALID_IMAGES, VALID_ANNOT,\
    TEST_IMAGES, TEST_ANNOT, TRAIN_OUTPUT, VALID_OUTPUT, TEST_OUTPUT


def class_text_to_int(label):
    if label == 'hand':
        return 1
    else:
        None

def parse_coordinates(annot_path):
    mat = loadmat(annot_path)
    coords = []
    index = 0
    for e in mat['boxes'][0]:
        box_cordinates = 0
        coords.append(list())  # new bounding box cordinates
        for value in e[0][0]:
            if box_cordinates > 3:
                break
            coords[index].append((value[0][0], value[0][1]))
            box_cordinates += 1
        index += 1
    return coords


def create_tf_example(name, image_dir, annot_dir):
    image_path = os.path.join(image_dir, name+'.jpg')
    annot_path = os.path.join(annot_dir, name+'.mat')
    annot_mat = parse_coordinates(annot_path)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    print (encoded_jpg)
    exit(1)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = name.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    label = 'hand'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for coord in annot_mat:

        x_max, x_min, y_max, y_min = 0, float('inf'), 0, float('inf')
        for y, x in coord:
            x_max, x_min = max(x, x_max), min(x, x_min)
            y_max, y_min = max(y, y_max), min(y, y_min)
        # normalized cordinates
        # box cordinates in faster rcnn uses 0 and 1 to define the position of the bounding boxes.
        # so if my value is greater than 1, select 1
        xmins.append(max(float(x_min) / width, 0.0))
        ymins.append(max(float(y_min) / height, 0.0))
        xmaxs.append(min(float(x_max) / width, 1.0))
        ymaxs.append(min(float(y_max) / height, 1.0))
        classes_text.append(label.encode('utf8'))
        classes.append(class_text_to_int(label))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(image_dir, annot_dir, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)

    for image in tqdm(os.listdir(image_dir)):
        if '.jpg' in image:
            name = image.split('.')[0]
            tf_example = create_tf_example(name, image_dir, annot_dir)
            writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':

    create_tf_record(TRAIN_IMAGES, TRAIN_ANNOT, TRAIN_OUTPUT)
    create_tf_record(VALID_IMAGES, VALID_ANNOT, VALID_OUTPUT)
    create_tf_record(TEST_IMAGES, TEST_ANNOT, TEST_OUTPUT)



