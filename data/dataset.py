import os


import tensorflow as tf

dataset_path = '/home/linze/liuhy/cnn_depth_tensorflow-master/cnn_depth_tensorflow-master/cnn_depth/data/dataset'


class Dataset(object):

  def __init__(self, task):
    #task : train, eval, test
    self.task = task



  def file_name(self):
    return os.path.join(dataset_path, '%s.csv' % self.task)

