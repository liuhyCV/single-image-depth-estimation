#encoding: utf-8
import os
import numpy as np
import h5py
from PIL import Image
import random
import png

import numpy as np
from skimage import io



def convert_nyu_norm(path):
    print("load dataset: %s" % (path))
    f = h5py.File(path)

    trains = []
    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):

        #set the first dimension of image/depth present the number of images 
        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)

        re_depth = (ra_depth/np.max(ra_depth))*255.0

        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join(code_directory, 'data', "nyu_datasets", "%05d_img.png" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join(code_directory, 'data', "nyu_datasets", "%05d_dep.png" % (i))
        depth_pil.save(depth_name)

        trains.append((image_name, depth_name))

        print(i)

    random.shuffle(trains)

    with open('train.csv', 'w') as output:
        for (image_name, depth_name) in trains:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")


def convert_nyu_16bit_image(path, target_dataset_directory):
    print("load dataset: %s" % (path))
    print("target dataset dir: %s" % (target_dataset_directory))

    f = h5py.File(path)

    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):

        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0) * 100

        re_depth = (ra_depth/np.max(ra_depth))*255.0

        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join(target_dataset_directory, "%05d_img.png" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join(target_dataset_directory, "%05d_dep.png" % (i))
        depth_pil.save(depth_name)

        depth_meters_name = os.path.join(target_dataset_directory, "%05d_dep_meters.png" % (i))
        with open(depth_meters_name, 'wb') as f:
            writer = png.Writer(width=ra_depth.shape[1], height=ra_depth.shape[0], bitdepth=16, greyscale=True)
            zgray2list = ra_depth.tolist()
            writer.write(f, zgray2list)

        print(i)


if __name__ == '__main__':

    target_dataset_directory = ( '/home/linze/liuhy/code/cnn_depth/data/dataset/NYU_dataset')
    dataset_directory = '/home/linze/liuhy/datasets/NYU_depth_v2_labeled'
    nyu_path = dataset_directory + '/nyu_depth_v2_labeled.mat'

    convert_nyu_16bit_image(nyu_path, target_dataset_directory)








