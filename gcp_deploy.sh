#!/bin/bash
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz

tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz

cp models/research/object_detection/samples/configs/faster_rcnn_resnet101_coco.config .

sed -i "" "s|PATH_TO_BE_CONFIGURED|"gs://$1"/data|g" faster_rcnn_resnet101_coco.config
sed -i "" "s|mscoco_label_map.pbtxt|label_map.pbtxt|g" faster_rcnn_resnet101_coco.config
sed -i "" "s|mscoco_train.record-?????-of-00100|train_tf_record.record|g" faster_rcnn_resnet101_coco.config
sed -i "" "s|mscoco_val.record-?????-of-00010|valid_tf_record.record|g" faster_rcnn_resnet101_coco.config
sed -i "" "s|num_classes: 90|num_classes: 1|g" faster_rcnn_resnet101_coco.config


cp $PWD/workspace/training_demo/data/hand_detection/training_dataset/training_data/train_tf_record.record .
cp $PWD/workspace/training_demo/data/hand_detection/test_dataset/test_data/test_tf_record.record . 
cp $PWD/workspace/training_demo/data/hand_detection/validation_dataset/validation_data/valid_tf_record.record .
cp $PWD/workspace/training_demo/data/label_map.pbtxt .


gsutil cp faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.* gs://$1/data/
gsutil cp faster_rcnn_resnet101_coco.config gs://$1/data/fast_rcnn_resnet101_coco.config

gsutil cp train_tf_record.record gs://$1/data/train_tf_record.record
gsutil cp test_tf_record.record gs://$1/data/test_tf_record.record
gsutil cp valid_tf_record.record gs://$1/data/valid_tf_record.record
gsutil cp label_map.pbtxt gs://$1/data/label_map.pbtxt

rm train_tf_record.record
rm test_tf_record.record
rm valid_tf_record.record
rm label_map.pbtxt
rm -r faster_rcnn_resnet101_coco.config
rm -r faster_rcnn_resnet101_coco_2018_01_28.tar.gz
rm -r faster_rcnn_resnet101_coco_2018_01_28
