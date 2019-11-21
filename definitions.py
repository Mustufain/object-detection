import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
DATA_DIR = 'workspace/training_demo/data/hand_detection'

TRAIN_IMAGES = os.path.join(ROOT_DIR, DATA_DIR, 'training_dataset', 'training_data', 'images')
TRAIN_ANNOT = os.path.join(ROOT_DIR, DATA_DIR, 'training_dataset', 'training_data', 'annotations')
TRAIN_OUTPUT = os.path.join(ROOT_DIR, DATA_DIR, 'training_dataset', 'training_data', 'train_tf_record.record')


VALID_IMAGES = os.path.join(ROOT_DIR, DATA_DIR, 'validation_dataset', 'validation_data', 'images')
VALID_ANNOT = os.path.join(ROOT_DIR, DATA_DIR, 'validation_dataset', 'validation_data', 'annotations')
VALID_OUTPUT = os.path.join(ROOT_DIR, DATA_DIR, 'validation_dataset', 'validation_data', 'valid_tf_record.record')

TEST_IMAGES = os.path.join(ROOT_DIR, DATA_DIR, 'test_dataset', 'test_data', 'images')
TEST_ANNOT = os.path.join(ROOT_DIR, DATA_DIR, 'test_dataset', 'test_data', 'annotations')
TEST_OUTPUT = os.path.join(ROOT_DIR, DATA_DIR, 'test_dataset', 'test_data', 'test_tf_record.record')

