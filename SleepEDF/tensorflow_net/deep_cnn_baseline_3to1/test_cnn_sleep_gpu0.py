import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from cnn_sleep_config import Config
from cnn_sleep import CNN_Sleep
from datagenerator_from_list_v2 import DataGenerator
from equaldatagenerator_from_list_v2 import EqualDataGenerator
from datagenerator_nchannel import DataGeneratorNChannel
from equaldatagenerator_nchannel import EqualDataGeneratorNChannel

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from scipy.io import loadmat,savemat


# Misc Parameters
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

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.app.flags.DEFINE_float("dropout_keep_prob_conv", 0.8, "Convolutional dropout keep probability (default: 0.8)")

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
config.dropout_keep_prob_conv = FLAGS.dropout_keep_prob_conv

eeg_active = ((FLAGS.eeg_train_data != "") & (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") & (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") & (FLAGS.emg_test_data != ""))

num_channel = 0

if (eeg_active):
    print("eeg active")
    num_channel += 1

    # Initalize the data generator seperately for the training, validation, and test sets
    eeg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.n_time, config.n_dim], shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.n_time, config.n_dim], shuffle = False, test_mode=True)
    # load pretrained filterbanks and do filtering first
    eeg_filter = loadmat(FLAGS.eeg_pretrainedfb_path)
    Wfb = eeg_filter['Wfb']
    eeg_train_gen.filter_with_filterbank(Wfb)
    eeg_test_gen.filter_with_filterbank(Wfb)
    del Wfb, eeg_filter

    # normalization here
    X = eeg_train_gen.X
    X = np.reshape(X,(eeg_train_gen.data_size*eeg_train_gen.data_shape[0], eeg_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)

    X = eeg_test_gen.X
    X = np.reshape(X,(eeg_test_gen.data_size*eeg_test_gen.data_shape[0], eeg_test_gen.data_shape[1]))
    X = (X - meanX) / stdX
    eeg_test_gen.X = np.reshape(X, (eeg_test_gen.data_size, eeg_test_gen.data_shape[0], eeg_test_gen.data_shape[1]))
    del X, eeg_train_gen


if (eog_active):
    print("eog active")
    num_channel += 1

    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.n_time, config.n_dim], shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.n_time, config.n_dim], shuffle = False, test_mode=True)
    # load pretrained filterbanks and do filtering first
    eog_filter = loadmat(FLAGS.eog_pretrainedfb_path)
    Wfb = eog_filter['Wfb']
    eog_train_gen.filter_with_filterbank(Wfb)
    eog_test_gen.filter_with_filterbank(Wfb)
    del Wfb, eog_filter

    # normalization here
    X = eog_train_gen.X
    X = np.reshape(X,(eog_train_gen.data_size*eog_train_gen.data_shape[0], eog_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)

    X = eog_test_gen.X
    X = np.reshape(X,(eog_test_gen.data_size*eog_test_gen.data_shape[0], eog_test_gen.data_shape[1]))
    X = (X - meanX) / stdX
    eog_test_gen.X = np.reshape(X, (eog_test_gen.data_size, eog_test_gen.data_shape[0], eog_test_gen.data_shape[1]))
    del X, eog_train_gen

if (emg_active):
    print("emg active")
    num_channel += 1

    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.n_time, config.n_dim], shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.n_time, config.n_dim], shuffle = False, test_mode=True)
    # load pretrained filterbanks and do filtering first
    emg_filter = loadmat(FLAGS.emg_pretrainedfb_path)
    Wfb = emg_filter['Wfb']
    emg_train_gen.filter_with_filterbank(Wfb)
    emg_test_gen.filter_with_filterbank(Wfb)
    del Wfb, emg_filter

    # normalization here
    X = emg_train_gen.X
    X = np.reshape(X,(emg_train_gen.data_size*emg_train_gen.data_shape[0], emg_train_gen.data_shape[1]))
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)

    X = emg_test_gen.X
    X = np.reshape(X,(emg_test_gen.data_size*emg_test_gen.data_shape[0], emg_test_gen.data_shape[1]))
    X = (X - meanX) / stdX
    emg_test_gen.X = np.reshape(X, (emg_test_gen.data_size, emg_test_gen.data_shape[0], emg_test_gen.data_shape[1]))
    del X, emg_train_gen

config.n_dim = eeg_train_gen.data_shape[1] # update the frequency dimension
config.n_channel = num_channel # update the number of channels

test_generator = DataGeneratorNChannel(data_shape=[config.n_time, config.n_dim, num_channel], shuffle = False, test_mode=True)

# expanding and concatenation
if(num_channel == 1):
    test_generator.X = np.expand_dims(eeg_test_gen.X,axis=3)
elif(num_channel == 2):
    test_generator.X = np.concatenate((np.expand_dims(eeg_test_gen.X,axis=3),
                                        np.expand_dims(eog_test_gen.X,axis=3)), axis=-1)
else: # num_channel == 3
    test_generator.X = np.concatenate((np.expand_dims(eeg_test_gen.X,axis=3),
                                        np.expand_dims(eog_test_gen.X,axis=3),
                                        np.expand_dims(emg_test_gen.X,axis=3)), axis=-1)
test_generator.y = eeg_test_gen.y
test_generator.label = eeg_test_gen.label
test_generator.boundary_index = eeg_test_gen.boundary_index

# clear individual data
del eeg_test_gen
if (eog_active):
    del eog_test_gen
if (emg_active):
    del emg_test_gen

# generate index here
test_generator.indexing()

config.n_time = test_generator.data_shape[0]*3 # update the time dimension as we used 3-epoch contextual input

test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.int16)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN_Sleep(config=config)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def dev_step(x_batch, y_batch):
            feed_dict = {
                cnn.X: x_batch,
                cnn.Y: y_batch,
                cnn.dropout_keep_prob: 1.0,
                cnn.dropout_keep_prob_conv: 1.0
            }
            _, loss, yhat, score, acc = sess.run(
                [global_step, cnn.loss, cnn.pred_Y, cnn.score, cnn.accuracy],
                feed_dict)
            return acc, yhat, score

        saver = tf.train.Saver(tf.all_variables())

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")

        test_yhat = np.zeros_like(test_generator.label)
        score = np.zeros([test_generator.data_size,config.n_class])
        test_step = 1
        while test_step < test_batches_per_epoch:
            x_batch, y_batch, _ = test_generator.next_batch(config.batch_size)
            _, test_yhat_, score_ = dev_step(x_batch, y_batch)
            test_yhat[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat_
            score[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score_
            test_step += 1
        if(test_generator.pointer < test_generator.data_size):
            actual_len, x_batch, y_batch, _ = test_generator.rest_batch(config.batch_size)
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
