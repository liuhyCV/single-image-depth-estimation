import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import re


IMAGE_HEIGHT = 172
IMAGE_WIDTH = 576
TARGET_HEIGHT = 27
TARGET_WIDTH = 142
'''
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74
'''

class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename, depthMeters_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"], ["meters"]])
        # input
        rgb_png = tf.read_file(filename)
        image = tf.image.decode_png(rgb_png, channels=3)
        image = tf.cast(image, tf.float32)       
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        #depth = tf.cast(depth, tf.int64)
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths

    def csv_inputs_test(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename, depthMeters_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"], ["meters"]])
        # input
        rgb_png = tf.read_file(filename)
        image = tf.image.decode_png(rgb_png, channels=3)
        image = tf.cast(image, tf.float32)
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths, filenames, depth_filenames = tf.train.batch(
            [image, depth, invalid_depth, filename, depth_filename],
            batch_size=self.batch_size,
            num_threads=4,
            capacity= 50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths, filenames, depth_filenames

    def csv_inputs_test_single(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        #print csv_file_path
        #test_which_dataset = re.sub(r'[/.]]', '', re.findall(r'/[a-zA-Z0-9]+[.]]', csv_file_path)[0])
        #print test_which_dataset

        if((csv_file_path == 'data/nyu_datasets.csv') | (csv_file_path == 'data/test_single.csv')):
            filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        elif((csv_file_path == 'data/test.csv') | (csv_file_path == 'data/train.csv') | (csv_file_path == 'data/vkitti_train.csv') | (csv_file_path == 'data/vkitti_test.csv')):
            filename, depth_filename, depth_meters_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"], ["depth_meters"]])

        # input
        rgb_png = tf.read_file(filename)
        image = tf.image.decode_png(rgb_png, channels=3)
        image = tf.cast(image, tf.float32)
        # target
        depth_png = tf.read_file(depth_filename)
        depth = tf.image.decode_png(depth_png, channels=1)
        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths, filenames, depth_filenames = tf.train.batch(
            [image, depth, invalid_depth, filename, depth_filename],
            batch_size=1,
            num_threads=1,
            capacity= 1,
        )
        return images, depths, invalid_depths, filenames, depth_filenames



def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d_dep.png" % (output_dir, i)
        depth_pil.save(depth_name)

def output_predict_test(true_depths, depths, images, filenames, depth_filenames, output_dir, current_test_number):

    #print images.shape

    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth, true_depth, filename) in enumerate(zip(images, depths, true_depths, filenames)):

        #print filenames
        img_info = re.sub(r'/', '_', re.findall(r'data/[a-zA-Z0-9_]+/[a-zA-Z0-9_]+/[a-zA-Z0-9]+', filename)[0])[0]

        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%s_org.png" % (output_dir, img_info)
        pilimg.save(image_name)

        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%s_dep.png" % (output_dir, img_info)
        depth_pil.save(depth_name)

        true_depth = true_depth.transpose(2, 0, 1)
        if np.max(true_depth) != 0:
            ra_true_depth = (true_depth/np.max(true_depth))*255.0
        else:
            ra_true_depth = true_depth*255.0
        true_depth_pil = Image.fromarray(np.uint8(ra_true_depth[0]), mode="L")
        true_depth_name = "%s/%s_ture.png" % (output_dir, img_info)
        true_depth_pil.save(true_depth_name)

def output_predict_test_single(true_depths, depths, images, filenames, depth_filenames, output_dir, current_test_number):

    #print images.shape

    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth, true_depth, filename) in enumerate(zip(images, depths, true_depths, filenames)):

        #print filenames
        print filename
        #img_info =  re.sub(r'/', '_', re.findall(r'data/[a-zA-Z0-9_]+/[a-zA-Z0-9_]+/[a-zA-Z0-9]+', filename)[0])
        #img_info = re.sub(r'[/]', '_', re.findall(r'data/[a-zA-Z0-9_/]+', filename)[0])
        img_info = re.sub(r'[/]', '_', re.findall(r'vkitti_1.3.1[a-zA-Z0-9_/]+', filename)[0])

        print img_info

        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%s_org.png" % (output_dir, img_info)
        pilimg.save(image_name)

        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%s_dep.png" % (output_dir, img_info)
        depth_pil.save(depth_name)

        true_depth = true_depth.transpose(2, 0, 1)
        if np.max(true_depth) != 0:
            ra_true_depth = (true_depth/np.max(true_depth))*255.0
        else:
            ra_true_depth = true_depth*255.0
        true_depth_pil = Image.fromarray(np.uint8(ra_true_depth[0]), mode="L")
        true_depth_name = "%s/%s_ture.png" % (output_dir, img_info)
        true_depth_pil.save(true_depth_name)
