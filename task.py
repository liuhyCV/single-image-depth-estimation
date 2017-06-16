#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict, output_predict_test, output_predict_test_single
import model
import train_operation as op

MAX_STEPS = 200
MAX_TEST_STEPS = 200/8

LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8

TRAIN_FILE = "data/vkitti_train.csv"
TEST_FILE = "data/vkitti_test.csv"
#SINGLE_TEST_FILE = "data/test.csv"
SINGLE_TEST_FILE = "data/vkitti_test.csv"
#SINGLE_TEST_FILE = "data/nyu_datasets.csv"
test_single_number = 500

COARSE_DIR = "output/coarse"
REFINE_DIR = "output/refine"
Summaries_DIR = "output/summary"

TRAIN = True
REFINE_TRAIN = False
FINE_TUNE = False

TEST = False
REFINE_TEST = False
TEST_SINGLE = True

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden)
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)


        loss = model.loss(logits, depths, invalid_depths)
        #tf.summary.scalar('loss', loss)

        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

        #merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(Summaries_DIR + '/train', sess.graph)

        sess.run(init_op)


        # parameters
        coarse_params = {}
        refine_params = {}
        if REFINE_TRAIN:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" %(variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        # define saver
        print '-----------------'
        print coarse_params
        print '-----------------'
        print refine_params
        print '-----------------'
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in xrange(MAX_STEPS):
            index = 0
            for i in xrange(1000):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images],
                                                                          feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    #train_writer.add_summary(summary, step*1000+i)
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value*100))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    if REFINE_TRAIN:
                        output_predict(logits_val, images_val, "output/predict/predict_refine_%05d_%05d" % (step, i))
                    else:
                        output_predict(logits_val, images_val, "output/predict/predict_%05d_%05d" % (step, i))
                index += 1

            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()

def test():
    with tf.Graph().as_default():
        dataset = DataSet(BATCH_SIZE)
        if(TEST_SINGLE):
            images, depths, invalid_depths, filenames, depth_filenames = dataset.csv_inputs_test_single(SINGLE_TEST_FILE)
        else:
            images, depths, invalid_depths, filenames, depth_filenames = dataset.csv_inputs_test(TEST_FILE)

        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        if REFINE_TEST:
            print("refine test.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv, keep_hidden, trainable=False)
        else:
            print("coarse test.")
            logits = model.inference(images, keep_conv, trainable=False)
        loss = model.loss(logits, depths, invalid_depths)
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        # parameters
        coarse_params = {}
        refine_params = {}
        if REFINE_TEST:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" % (variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable

        # define saver
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TEST:
            saver_refine = tf.train.Saver(refine_params)

        # load parameters
        coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        else:
            print("No Pretrained coarse Model.")
        if REFINE_TEST:
            refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
            if refine_ckpt and refine_ckpt.model_checkpoint_path:
                print("Pretrained refine Model Loading.")
                saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                print("Pretrained refine Model Restored.")
            else:
                print("No Pretrained refine Model.")

        # test

        #Start populating the filename queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if(TEST_SINGLE):
            MAX_TEST_STEPS = test_single_number

        for step in xrange(MAX_TEST_STEPS):

            ture_depths, logits_val, images_val, loss_val, filenames_val, depth_filenames_val = sess.run([depths, logits, images, loss, filenames, depth_filenames], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})

            if REFINE_TEST:
                if(TEST_SINGLE):
                    output_predict_test_single(ture_depths, logits_val, images_val, filenames_val, depth_filenames_val,  "output/test/single_test_refine", step * BATCH_SIZE)
                else:
                    output_predict_test(ture_depths, logits_val, images_val, filenames_val, depth_filenames_val,  "output/test/test_refine", step * BATCH_SIZE)
            else:
                if(TEST_SINGLE):
                    output_predict_test_single(ture_depths, logits_val, images_val, filenames_val, depth_filenames_val,  "output/test/single_test", step* BATCH_SIZE)
                else:
                    output_predict_test(ture_depths, logits_val, images_val, filenames_val, depth_filenames_val,  "output/test/test", step* BATCH_SIZE)

            print("%s: %d[step]: test loss %f" % (datetime.now(), step, loss_val))
            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        coord.request_stop()
        coord.join(threads)
        sess.close()



def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)

    if(TEST):
        test()
    elif(TRAIN):
        train()


if __name__ == '__main__':
    tf.app.run()
