import requests
import base64
import json
import cv2
import numpy as np
import ast
from argparse import ArgumentParser

# Read args
parser = ArgumentParser(description='Send REST request to classify image of fungi.')
parser.add_argument('--image',
                    dest="image",
                    default='./images/boletus.jpg',
                    help='path to fungi image to be classified',)
parser.add_argument('--host',
                    dest="host",
                    default='localhost',
                    help='server address for predictions',)
args = parser.parse_args()

# Read type names
classes = "./classes"
class_idx = []
with open(classes, 'r') as f:
    for line in f:
        _, name = line.split(' ', 1)
        class_idx.append(name[:-1])
assert len(class_idx) == 1394, "Unexpected number of classes"

# Create request with the selected image
URL = "http://{}:8501/v1/models/model:predict".format(args.host)
headers = {"content-type": "application/json"}
image_content = base64.b64encode(open(args.image,'rb').read()).decode()
body = {
    "instances": [ {
        "image_in": {
            "b64": image_content
        }
    } ]
}
json_dump = json.dumps(body)
json_dump = bytearray(json_dump, 'utf-8')

# Send request, get result
r = requests.post(URL, data=json_dump, headers = headers)
x = ast.literal_eval(r.text)

# Print sorted types and their probabilities
probabilities = np.array(x['predictions'][0])
ind = probabilities.argsort()[-5:][::-1]
# faster approach but doesn't give sorted values
# ind = np.argpartition(probabilities, -5)[-5:]#[::-1]
print('Predicted classes:')
print([str(probabilities[i]) + ': ' + class_idx[i] for i in ind])
