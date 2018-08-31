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

config.n_dim = eeg_test_gen.data_shape[1] # update frequency dimension
config.n_channel = num_channel # update number of channels

test_generator = DataGeneratorNChannel(data_shape=[config.n_time, config.n_dim, num_channel], shuffle = False, test_mode=True)

# expanding channel dimension and concatenation
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

        def dev_step(x_batch, y_batch1, y_batch2, y_batch3):
            feed_dict = {
                cnn.X: x_batch,
                cnn.Y1: y_batch1,
                cnn.Y2: y_batch2,
                cnn.Y3: y_batch3,
                cnn.dropout_keep_prob: 1.0,
                cnn.dropout_keep_prob_conv: 1.0
            }
            _, loss, yhat1, score1, acc1, \
                    yhat2, score2, acc2, \
                    yhat3, score3, acc3 = sess.run(
                [global_step, cnn.loss, cnn.pred_Y1, cnn.score1, cnn.accuracy1,
                 cnn.pred_Y2, cnn.score2, cnn.accuracy2,
                 cnn.pred_Y3, cnn.score3, cnn.accuracy3],
                feed_dict)
            return acc1, acc2, acc3, yhat1, yhat2, yhat3, score1, score2, score3

        saver = tf.train.Saver(tf.all_variables())

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")

        test_yhat1 = np.zeros_like(test_generator.data_index)
        score1 = np.zeros([len(test_generator.data_index),config.n_class])
        test_yhat2 = np.zeros_like(test_generator.data_index)
        score2 = np.zeros([len(test_generator.data_index),config.n_class])
        test_yhat3 = np.zeros_like(test_generator.data_index)
        score3 = np.zeros([len(test_generator.data_index),config.n_class])
        test_step = 1
        while test_step < test_batches_per_epoch:
            x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = test_generator.next_batch(config.batch_size)
            _, _, _, test_yhat1_, test_yhat2_, test_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            test_yhat1[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat1_
            score1[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score1_
            test_yhat2[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat2_
            score2[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score2_
            test_yhat3[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat3_
            score3[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score3_
            test_step += 1
        if(test_generator.pointer < len(test_generator.data_index)):
            actual_len, x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = test_generator.rest_batch(config.batch_size)
            _, _, _, test_yhat1_, test_yhat2_, test_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            test_yhat1[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat1_
            score1[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score1_
            test_yhat2[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat2_
            score2[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score2_
            test_yhat3[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat3_
            score3[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score3_
        test_yhat1 = test_yhat1 + 1
        test_yhat2 = test_yhat2 + 1
        test_yhat3 = test_yhat3 + 1
        test_fscore1 = f1_score(test_generator.label[:-1], test_yhat1[1:], average='macro')
        test_acc1 = accuracy_score(test_generator.label[:-1], test_yhat1[1:])
        test_kappa1 = cohen_kappa_score(test_generator.label[:-1], test_yhat1[1:])
        test_fscore2 = f1_score(test_generator.label, test_yhat2, average='macro')
        test_acc2 = accuracy_score(test_generator.label, test_yhat2)
        test_kappa2 = cohen_kappa_score(test_generator.label, test_yhat2)
        test_fscore3 = f1_score(test_generator.label[1:], test_yhat3[:-1], average='macro')
        test_acc3 = accuracy_score(test_generator.label[1:], test_yhat3[:-1])
        test_kappa3 = cohen_kappa_score(test_generator.label[1:], test_yhat3[:-1])
        savemat(os.path.join(out_path, "test_ret_model_acc.mat"), dict(acc1=test_acc1,
                                                                       fscore1=test_fscore1,
                                                                       kappa1=test_kappa1,
                                                                       acc2=test_acc2,
                                                                       fscore2=test_fscore2,
                                                                       kappa2=test_kappa2,
                                                                       acc3=test_acc3,
                                                                       fscore3=test_fscore3,
                                                                       kappa3=test_kappa3,
                                                                       yhat1=test_yhat1,
                                                                       yhat2=test_yhat2,
                                                                       yhat3=test_yhat3,
                                                                       score1 = score1,
                                                                       score2 = score2,
                                                                       score3 = score3))
        test_generator.reset_pointer()
