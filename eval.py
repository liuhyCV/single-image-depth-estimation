import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage import io
'''
im = plt.imread("output.png")

print im.shape

np.save('im.npy',im)

'''

depth_png = tf.read_file('foo_gray.png')

depth = tf.image.decode_png(depth_png, dtype=tf.uint16)
depth_16 = tf.cast(depth, tf.uint16)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)





with tf.Session() as sess:
    tf_depth_png = depth_png.eval()
    #print tf_depth_png
    k_data = depth.eval()
    #print k_data.shape
    data = depth_16.eval()
    print data
    cv2.imshow(data)
    cv2.waitKey()