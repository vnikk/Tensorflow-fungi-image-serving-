# Description
This module takes a trained InceptionV4/InceptionResnetV2 model trained on FGVCx Fungi dataset, conerts it to servable format and exposes using TF Serving.

## Environment
Conversion script requires TensorFlow v1.4 (unfortunately for now).

For singularity script user needs to provide `MODEL_PATH` and `SINGULARITY_SERVING_IMAGE` variables. 

## Running
Model can be converted using `convert_model_for_serving.py`. Aside from using default values, user can specify model export directory, desired model type (InceptionV4 / InceptionResnetV2) and graph definition path.

When the model is exported it can be executed with TF Serving using `run_model_in_singularity.sh`, where user also needs to uncomment the model type to be run.

Querying the prediction is done with `serving_request.py`.
There is one testing image in `./images`, but you can provide your own with `--image` flag to the script.
Fungi types are read from `classes` file. If model's output vector is changed, this file would need adjustment too.

## Versions
Running the conversion script:
TensorFlow - v1.4

Serving:
Singularity - 2.5.2-dist
Serving image - tensorflow_serving_1.13.0_cpu.simg
