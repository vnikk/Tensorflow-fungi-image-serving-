#!/usr/bin/env sh

# newly frozen inception_resnet_v2
model_name=frozen_inception_resnet_v2_fungi2018.pb
# newly frozen inception_v4
#model_name=frozen_inception_v4_fungi2018.pb

echo "Using model from: $MODEL_PATH/$model_name"
ls "$MODEL_PATH/$model_name"
SINGULARITYENV_MODEL_NAME=$model_name singularity run --nv -B $MODEL_PATH/$model_name:/models/model $SINGULARITY_SERVING_IMAGE
