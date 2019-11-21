cd models/research
JOB_NAME=object_detection"_$(date +%m_%d_%Y_%H_%M_%S)"
echo $JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
	--job-dir=gs://$1/train \
	--scale-tier BASIC_GPU \
	--runtime-version 1.12 \
	--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
	--module-name object_detection.model_main \
	--region europe-west1 \
	-- \
	--model_dir=gs://$1/train \
	--pipeline_config_path=gs://$1/data/fast_rcnn_resnet101_coco.config	 

