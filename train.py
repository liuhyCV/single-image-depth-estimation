import copy
from datetime import datetime
import os.path
import re
import time
from tensorflow.python.platform import gfile
from PIL import Image

import numpy as np
import tensorflow as tf
import argparse

from model.model import inference
from model.model import loss
from image_processing import train_batch_inputs
from image_processing import eval_batch_inputs

class Train_Flags():
    def __init__(self):
        self.max_step = 100000
        self.num_per_epoch = 1000
        self.num_epochs_per_decay = 30
        self.batch_size = 10
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999

        self.dataset_train_csv_file_path = '/home/linze/liuhy/code/cnn_depth/data/dataset/train.csv'
        self.dataset_eval_csv_file_path = '/home/linze/liuhy/code/cnn_depth/data/dataset/eval.csv'
        self.eval_num = 280

        self.output_summary_path = '/home/linze/liuhy/code/cnn_depth/result/summary'
        self.output_check_point_path = '/home/linze/liuhy/code/cnn_depth/result/check_point'
        self.output_train_predict_depth_path = '/home/linze/liuhy/code/cnn_depth/result/network_predict/train_predict'
        self.output_eval_predict_depth_path = '/home/linze/liuhy/code/cnn_depth/result/network_predict/eval_predict'
        self.output_test_predict_depth_path = '/home/linze/liuhy/code/cnn_depth/result/network_predict/test_predict'


        self.check_path_exist()


    def check_path_exist(self):
        if not gfile.Exists(self.output_summary_path):
            gfile.MakeDirs(self.output_summary_path)
        if not gfile.Exists(self.output_check_point_path):
            gfile.MakeDirs(self.output_check_point_path)

        if not gfile.Exists(self.output_train_predict_depth_path):
            gfile.MakeDirs(self.output_train_predict_depth_path)
        if not gfile.Exists(self.output_eval_predict_depth_path):
            gfile.MakeDirs(self.output_eval_predict_depth_path)
        if not gfile.Exists(self.output_test_predict_depth_path):
            gfile.MakeDirs(self.output_test_predict_depth_path)


train_flags = Train_Flags()


def _model_loss(images, depths, invalid_depths, mode):
    # Compute the moving average of all losses

    flag_reuse_train_eval = tf.placeholder(tf.bool)
    flag_trainable_train_eval = tf.placeholder(tf.bool)


    if (mode == 'train'):
        flag_reuse_train_eval = False
        flag_trainable_train_eval = True
    elif (mode == 'eval'):
        flag_reuse_train_eval = True
        flag_trainable_train_eval = False


    with tf.variable_scope(tf.get_variable_scope()):
            logits = inference(images, reuse=flag_reuse_train_eval, trainable=flag_trainable_train_eval)

    loss(logits, depths, invalid_depths)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Compute the moving average of total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss, logits


def save(images, depths, predict_depths, global_step, target_path, batch_number=None, mode='train'):

    output_dir = os.path.join(target_path, str(global_step))

    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth, predict_depth) in enumerate(zip(images, depths, predict_depths)):
        if(batch_number == None):
            image_name = "%s/%05d_rgb.png" % (output_dir, i)
            depth_name = "%s/%05d_depth.png" % (output_dir, i)
            predict_depth_name = "%s/%05d_predict.png" % (output_dir, i)
        else:
            image_name = "%s/%d_%05d_rgb.png" % (output_dir, batch_number, i)
            depth_name = "%s/%d_%05d_depth.png" % (output_dir, batch_number, i)
            predict_depth_name = "%s/%d_%05d_predict.png" % (output_dir, batch_number, i)


        pilimg = Image.fromarray(np.uint8(image))
        pilimg.save(image_name)

        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_pil.save(depth_name)

        predict_depth = predict_depth.transpose(2, 0, 1)
        if np.max(predict_depth) != 0:
            ra_depth = (predict_depth/np.max(predict_depth))*255.0
        else:
            ra_depth = predict_depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_pil.save(predict_depth_name)



def train():
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default():
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)


        eval_average_loss = tf.get_variable(
            'eval_average_loss', [], trainable=False)

        tf.summary.scalar('eval_average_loss', eval_average_loss)


        # Calculate the learning rate schedule.
        num_batches_per_epoch = (train_flags.num_per_epoch /
                                 train_flags.batch_size)

        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)


        images, depths, invalid_depths = train_batch_inputs(train_flags.dataset_train_csv_file_path, train_flags.batch_size)

        eval_images, eval_depths, eval_invalid_depths = eval_batch_inputs(train_flags.dataset_eval_csv_file_path, train_flags.batch_size)

        loss, logits_inference = _model_loss(images, depths, invalid_depths, mode='Train')
        tf.summary.scalar('loss', loss)


        eval_loss, eval_logits_inference = _model_loss(eval_images, eval_depths, eval_invalid_depths, mode='eval')


        # Create an optimizer that performs gradient descent.

        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(loss)
        # Calculate the gradients for each model tower.

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(train_flags.moving_average_decay, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # load restore parameters
        coarse_params = {}

        for variable in tf.trainable_variables():
            variable_name = variable.name
            print("parameter: %s" % (variable_name))
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)

        coarse_ckpt = tf.train.get_checkpoint_state(train_flags.output_check_point_path)
        if coarse_ckpt:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        else:
            print("No Pretrained coarse Model.")

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(train_flags.output_summary_path, graph=sess.graph)

        for step in range(train_flags.max_step):
            start_time = time.time()
            _, loss_value, images_batch, depths_batch, predict_depths = sess.run([train_op, loss, images, depths, logits_inference])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 100 == 0:

                eval_total_loss = 0
                eval_total_num = 0
                while (eval_total_num < train_flags.eval_num):
                    eval_loss_value, eval_images_batch, eval_depths_batch, eval_predict_depths = sess.run(
                        [eval_loss, eval_images, eval_depths, eval_logits_inference])
                    eval_total_loss = eval_total_loss + eval_loss_value
                    eval_total_num = eval_total_num + train_flags.batch_size
                eval_average_loss_out = sess.run(
                    [eval_average_loss], feed_dict={eval_average_loss:eval_total_loss / float(eval_total_num/train_flags.batch_size)})
                print("%d step: eval_average_loss_out: %.2f" % (step, eval_average_loss_out[0]))

                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                eval_loss_value, eval_images_batch, eval_depths_batch, eval_predict_depths = sess.run(
                    [eval_loss, eval_images, eval_depths, eval_logits_inference])

                save(eval_images_batch, eval_depths_batch, eval_predict_depths, step,
                     train_flags.output_eval_predict_depth_path)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == train_flags.max_step:
                checkpoint_path = os.path.join(train_flags.output_check_point_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                save(images_batch, depths_batch, predict_depths, step, train_flags.output_train_predict_depth_path)




        coord.request_stop()
        coord.join(threads)
        sess.close()
