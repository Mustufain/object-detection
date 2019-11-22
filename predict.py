from googleapiclient.discovery import  build
import os
import logging 
import base64
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def predict_json(project, model, request, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  #<path_to_service_account_file>
    service = build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)
    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body=request
    )
    response.execute()
    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def parse_response(response):
  boxes = response['predictions']['detection_boxes']
  return boxes



with tf.gfile.GFile('test_image/test_3.jpg', 'rb') as fid:
    encoded_jpg = base64.b64encode(fid.read())
image_base64 = encoded_jpg.decode()
request = {"instances": [{"image_bytes": {"b64": image_base64}}]}
prediction = predict_json('handdetector', 'fastercnn', request)
print (prediction)

