import tensorflow as tf

from dataset import Dataset

import scipy.io as sio


def train_image(dataset, batch_size=None):

    filename_queue = tf.train.string_input_producer([dataset.file_name()], shuffle=True)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    rgb_filename, depth_filename = tf.decode_csv(serialized_example,
                                                                   [["path"], ["meters"]])
    # input
    rgb_png = tf.read_file(rgb_filename)
    image = tf.image.decode_png(rgb_png, channels=3)
    image = tf.cast(image, tf.float32)

    # target
    depth_png = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth_png, channels=1)
    depth = tf.cast(depth, tf.float32)
    depth = tf.div(depth, [255.0])
    # depth = tf.cast(depth, tf.int64)
    # resize
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
    invalid_depth = tf.sign(depth)
    # generate batch
    images, depths, invalid_depths = tf.train.batch(
        [image, depth, invalid_depth],
        batch_size=self.batch_size,
        num_threads=4,
        capacity=50 + 3 * self.batch_size,
    )
    return images, depths, invalid_depths


def test_image(dataset, batch_size=None):
    pass


def val_iamge(dataset, batch_size=None):
    pass



