FROM ubuntu:16.04

# FROM python:3.6-buster
MAINTAINER Mustufain Rizvi <abbasmustufain@gmail.com>

## ENV Variables
ENV PYTHON_VERSION="3.6.5"
ENV BUCKET_NAME='detection-sandbox'
ENV DIRECTORY='/usr/local/gcloud'

# Update and Install packages
RUN apt-get update -y \
 && apt-get install -y \
 curl \
 wget \
 tar \
 xz-utils \
 bc \
 build-essential \
 cmake \
 curl \
 zlib1g-dev \
 libssl-dev \
 libsqlite3-dev \
 python3-pip \
 python3-setuptools \
 unzip \
 g++ \
 git \
 python-tk

 # Install Python 3.6.5
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar -xvf Python-${PYTHON_VERSION}.tar.xz \
    && rm -rf Python-${PYTHON_VERSION}.tar.xz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure \
    && make install \
    && cd / \
    && rm -rf Python-${PYTHON_VERSION}

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

## Download and Install Google Cloud SDK
RUN mkdir -p /usr/local/gcloud \
    && curl https://sdk.cloud.google.com > install.sh \
    && bash install.sh --disable-prompts --install-dir=${DIRECTORY}

# Adding the package path to directory
ENV PATH $PATH:${DIRECTORY}/google-cloud-sdk/bin

# working directory
WORKDIR /usr/src/app

COPY requirements.txt ./ \
    testproject-264512-9de8b1b35153.json ./

ENV GOOGLE_APPLICATION_CREDENTIALS=testproject-264512-9de8b1b35153.json

# PIP independencies
RUN pip --no-cache-dir install -r requirements.txt

# Install tensorflow object detection dependencies
RUN apt-get update \
    && apt-get install -y \
    protobuf-compiler \
    python-pil \
    python-lxml

# Clone tensorflow models directory and install Tensorflow object detection API
# Run tests to ernsure efficient installation of Tensorflow object detection API
RUN mkdir -p models/ \
    && git clone https://github.com/tensorflow/models.git /usr/src/app/models/ \
    && cd /usr/src/app/models/research/ \
    && wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip \
    && unzip protobuf.zip \
    && ./bin/protoc object_detection/protos/*.proto --python_out=. \
    && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim \
    #&& bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools \
    #&& python setup.py sdist \
    #&& cd slim \
    #&& python setup.py sdist \
    && python3 object_detection/builders/model_builder_test.py

# Google Cloud
RUN gcloud config set account docker-sandbox@testproject-264512.iam.gserviceaccount.com \
    && gcloud auth activate-service-account --key-file=testproject-264512-9de8b1b35153.json \
    && export GOOGLE_APPLICATION_CREDENTIALS=testproject-264512-9de8b1b35153.json \
    && gcloud config set project testproject-264512

# Create bucket, deploy file to ${BUCKET_NAME} and submit training job
RUN gsutil mb -l europe-north1 gs://${BUCKET_NAME}/ \
    && bash gcp_deploy.sh ${BUCKET_NAME}



ENTRYPOINT ["sleep"]

CMD [ "200" ]



###
# Before  creating bucket ,
# (Think of ways to automate it.)
# give role to your service account --> Stroage admin (full access to GCS)
# give role to your service account --> ML Engine admin (full access to Google AI Platform)
###clea