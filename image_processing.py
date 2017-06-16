import tensorflow as tf
import os.path

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74




def train_batch_inputs(dataset_csv_file_path, batch_size):

    with tf.name_scope('batch_processing'):

        if (os.path.isfile(dataset_csv_file_path) != True):
            raise ValueError('No data files found for this dataset')

        filename_queue = tf.train.string_input_producer([dataset_csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])

        # input
        png = tf.read_file(filename)
        image = tf.image.decode_png(png, channels=3)
        image = tf.cast(image, tf.float32)
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, dtype=tf.uint16, channels=1)
        depth = tf.cast(depth, dtype=tf.int16)

        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)

        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size = batch_size,
            num_threads = 4,
            capacity = 50 + 3 * batch_size
        )
        return images, depths, invalid_depths

def eval_batch_inputs(dataset_csv_file_path, batch_size):

    with tf.name_scope('eval_batch_processing'):

        if (os.path.isfile(dataset_csv_file_path) != True):
            raise ValueError('No data files found for this dataset')

        filename_queue = tf.train.string_input_producer([dataset_csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])

        # input
        png = tf.read_file(filename)
        image = tf.image.decode_png(png, channels=3)
        image = tf.cast(image, tf.float32)
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, dtype=tf.uint16, channels=1)
        depth = tf.cast(depth, dtype=tf.int16)

        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)

        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size = batch_size,
            num_threads = 4,
            capacity = 50 + 3 * batch_size
        )
        return images, depths, invalid_depths


