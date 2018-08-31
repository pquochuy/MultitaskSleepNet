#! /usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from datetime import datetime
import h5py
from scipy.io import loadmat,savemat

from cnn_1d_sleep import CNN1DSleep
from cnn_1d_sleep_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator


# Parameters
# ==================================================

tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_filter", 400, "Number of filters per filter size (default: 400)")

tf.app.flags.DEFINE_string("eeg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("eog_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("emg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))

config = Config()
config.dropout_keep_prob = FLAGS.dropout_keep_prob
config.num_filters = FLAGS.num_filter

eeg_active = ((FLAGS.eeg_train_data != "") & (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") & (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") & (FLAGS.emg_test_data != ""))

if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, test sets
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = True)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False)

    # load pretrained filterbank and do filtering first
    eeg_filter = loadmat(FLAGS.eeg_pretrainedfb_path)
    Wfb = eeg_filter['Wfb']
    eeg_train_gen.filter_with_filterbank(Wfb)
    eeg_test_gen.filter_with_filterbank(Wfb)
    del Wfb

if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.time_length, config.freq_length], shuffle = True)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False)

    # load pretrained filterbank and do filtering first
    eog_filter = loadmat(FLAGS.eog_pretrainedfb_path)
    Wfb = eog_filter['Wfb']
    eog_train_gen.filter_with_filterbank(Wfb)
    eog_test_gen.filter_with_filterbank(Wfb)
    del Wfb

if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = True)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False)

    # load pretrained filterbank and do filtering first
    emg_filter = loadmat(FLAGS.emg_pretrainedfb_path)
    Wfb = emg_filter['Wfb']
    emg_train_gen.filter_with_filterbank(Wfb)
    emg_test_gen.filter_with_filterbank(Wfb)
    del Wfb

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen

if (eog_active):
    train_generator.X = np.concatenate((train_generator.X, eog_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, eog_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]

if (emg_active):
    train_generator.X = np.concatenate((train_generator.X, emg_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, emg_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]


del eeg_train_gen
del eeg_test_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen


# data normalization here
X = train_generator.X
X = np.reshape(X,(train_generator.data_size*train_generator.data_shape[0], train_generator.data_shape[1]))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X = (X - meanX) / stdX
train_generator.X = np.reshape(X, (train_generator.data_size, train_generator.data_shape[0], train_generator.data_shape[1]))

X = test_generator.X
X = np.reshape(X,(test_generator.data_size*test_generator.data_shape[0], test_generator.data_shape[1]))
X = (X - meanX) / stdX
test_generator.X = np.reshape(X, (test_generator.data_size, test_generator.data_shape[0], test_generator.data_shape[1]))

config.freq_length = train_generator.data_shape[1]

test_batches_per_epoch = np.floor(test_generator.data_size / config.batch_size).astype(np.int16)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN1DSleep(
            time_length=config.time_length,
            freq_length=config.freq_length,
            num_classes=config.num_classes,
            filter_sizes=list(map(int, config.filter_sizes.split(","))),
            num_filters=config.num_filters,
            l2_reg_lambda=config.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def dev_step(x_batch, y_batch):
            feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
            step, loss, accuracy, pred_Y, score_ = sess.run(
                   [global_step, cnn.loss, cnn.accuracy, cnn.predictions, cnn.scores],
                   feed_dict)
            return accuracy, pred_Y, score_

        saver = tf.train.Saver(tf.all_variables())

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")


        test_yhat = np.zeros_like(test_generator.label)
        score = np.zeros([test_generator.data_size,config.num_classes])
        test_step = 1
        while test_step < test_batches_per_epoch:
            x_batch, y_batch, _ = test_generator.next_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, test_yhat_, score_ = dev_step(x_batch, y_batch)
            test_yhat[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat_
            score[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score_
            test_step += 1
        if(test_generator.pointer < test_generator.data_size):
            actual_len, x_batch, y_batch, _ = test_generator.rest_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, test_yhat_, score_ = dev_step(x_batch, y_batch)
            test_yhat[(test_step-1)*config.batch_size : test_generator.data_size] = test_yhat_
            score[(test_step-1)*config.batch_size : test_generator.data_size,:] = score_
        test_yhat = test_yhat + 1
        test_fscore = f1_score(test_generator.label, test_yhat, average='macro')
        test_acc = accuracy_score(test_generator.label, test_yhat)
        test_kappa = cohen_kappa_score(test_generator.label, test_yhat)
        savemat(os.path.join(out_path, "test_ret_model_acc.mat"), dict(acc=test_acc,
                                                                       fscore=test_fscore,
                                                                       kappa=test_kappa,
                                                                       yhat=test_yhat,
                                                                       score = score))
        test_generator.reset_pointer()
