import os
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def parse_args():
    parser = ArgumentParser(description='Convert the TensorFlow Inception/Resnet fungi models for serving.')
    parser.add_argument('--dir',
                        dest="export_dir",
                        help='path to export model to',)
    parser.add_argument('--type',
                        dest='model_type',
                        choices=['resnet', 'inception'],
                        default='resnet',
                        help='type of model to be converted')
    parser.add_argument('--graph',
                        dest='graph_file',
                        help='graph def file for the selected model')
    args = parser.parse_args()
    return args.export_dir, args.model_type, args.graph_file


def decode_and_resize(image_str_tensor):
    """Decodes jpeg string, resizes it and returns a uint8 tensor."""
    HEIGHT, WIDTH, CHANNELS = 299, 299, 3
    image = tf.image.decode_jpeg(image_str_tensor, channels=CHANNELS)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [HEIGHT, WIDTH], align_corners=False)
    image = tf.squeeze(image, squeeze_dims=[0])
    image = tf.cast(image, dtype=tf.uint8)
    return image


if __name__ == '__main__':
    export_dir, model_type, graph_file = parse_args()
    # TF Serving selects last version of the model from the model directory
    last_digit = sorted([int(i) for i in os.listdir(export_dir)])[-1]
    export_dir = os.path.join(export_dir, str(last_digit + 1))
    print('Writing model to {}'.format(export_dir))
    if model_type == 'inception':
        output_tensor_name = "InceptionV4/Logits/Predictions:0"
    else:
        output_tensor_name = "InceptionResnetV2/Logits/Predictions:0"

    sigs = {}
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    # Load graph definition
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Makes additional tensors to convert input and saves model ready for serving
    with tf.Session(graph=tf.Graph()) as sess:
        # placeholder for image in incoded string format
        input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
        images_tensor = tf.map_fn(decode_and_resize, input_ph, back_prop=False, dtype=tf.uint8)
        images_tensor = tf.image.convert_image_dtype(images_tensor, dtype=tf.float32)

        # name="" prevents spurious prefixing;
        # input_map prepends string tensor to input tensor
        tf.import_graph_def(graph_def, name="", input_map={"input:0": images_tensor})
        g = tf.get_default_graph()
        # Loads placeholder defined above and tensors applied to it to be used in serving
        image_tensor = g.get_tensor_by_name("image_binary:0")
        output_tensor = g.get_tensor_by_name(output_tensor_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"image_in": image_tensor},
                {"out": output_tensor})

        builder.add_meta_graph_and_variables(
                sess,
                [tag_constants.SERVING],
                signature_def_map=sigs)

    builder.save()
    print('Model saved to: ', export_dir)
