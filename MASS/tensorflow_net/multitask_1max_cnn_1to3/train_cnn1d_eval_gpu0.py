import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from cnn_1d_sleep import CNN1DSleep
from cnn_1d_sleep_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator
from equaldatagenerator_from_list_v2 import EqualDataGenerator

from scipy.io import loadmat

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_filter", 500, "Number of filters per filter size (default: 400)")

tf.app.flags.DEFINE_string("eeg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("eog_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("emg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout_keep_prob = FLAGS.dropout_keep_prob
config.num_filters = FLAGS.num_filter

eeg_active = ((FLAGS.eeg_train_data != "") & (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") & (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") & (FLAGS.emg_test_data != ""))

# it is important not to shuffle the data here as we are going to combine them after filtering
if (eeg_active):
    print("eeg active")
    # Initalize the data generator separately for the training, validation, and test sets
    eeg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    # load pretrained filterbank and do filtering first
    eeg_filter = loadmat(FLAGS.eeg_pretrainedfb_path)
    Wfb = eeg_filter['Wfb']
    eeg_train_gen.filter_with_filterbank(Wfb)
    eeg_test_gen.filter_with_filterbank(Wfb)
    eeg_eval_gen.filter_with_filterbank(Wfb)
    del Wfb, eeg_filter

if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    # load pretrained filterbank and do filtering first
    eog_filter = loadmat(FLAGS.eog_pretrainedfb_path)
    Wfb = eog_filter['Wfb']
    eog_train_gen.filter_with_filterbank(Wfb)
    eog_test_gen.filter_with_filterbank(Wfb)
    eog_eval_gen.filter_with_filterbank(Wfb)
    del Wfb, eog_filter

# EMG not active for Sleep-EDF
if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    # load pretrained filterbank and do filtering first
    emg_filter = loadmat(FLAGS.emg_pretrainedfb_path)
    Wfb = emg_filter['Wfb']
    emg_train_gen.filter_with_filterbank(Wfb)
    emg_test_gen.filter_with_filterbank(Wfb)
    emg_eval_gen.filter_with_filterbank(Wfb)
    del Wfb, emg_filter

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen
eval_generator = eeg_eval_gen

# concatenate different channels
if (eog_active):
    train_generator.X = np.concatenate((train_generator.X, eog_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, eog_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.concatenate((eval_generator.X, eog_eval_gen.X), axis=-1)
    eval_generator.data_shape = eval_generator.X.shape[1:]

if (emg_active):
    train_generator.X = np.concatenate((train_generator.X, emg_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, emg_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.concatenate((eval_generator.X, emg_eval_gen.X), axis=-1)
    eval_generator.data_shape = eval_generator.X.shape[1:]

del eeg_train_gen
del eeg_test_gen
del eeg_eval_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
    del eog_eval_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen
    del emg_eval_gen


# data normalization here
X = train_generator.X
X = np.reshape(X,(train_generator.data_size*train_generator.data_shape[0], train_generator.data_shape[1]))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X = (X - meanX) / stdX
train_generator.X = np.reshape(X, (train_generator.data_size, train_generator.data_shape[0], train_generator.data_shape[1]))
train_generator.shuffle_data()

X = eval_generator.X
X = np.reshape(X,(eval_generator.data_size*eval_generator.data_shape[0], eval_generator.data_shape[1]))
X = (X - meanX) / stdX
eval_generator.X = np.reshape(X, (eval_generator.data_size, eval_generator.data_shape[0], eval_generator.data_shape[1]))

X = test_generator.X
X = np.reshape(X,(test_generator.data_size*test_generator.data_shape[0], test_generator.data_shape[1]))
X = (X - meanX) / stdX
test_generator.X = np.reshape(X, (test_generator.data_size, test_generator.data_shape[0], test_generator.data_shape[1]))

# update the frequency dimension after filtering
config.freq_length = train_generator.data_shape[1]

train_batches_per_epoch = np.floor((train_generator.data_size - len(train_generator.boundary_index)) / config.batch_size).astype(np.int16)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.int16)

print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(train_generator.data_size, eval_generator.data_size, test_generator.data_size))

print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

# variable to keep track of best fscore
best_fscore = 0.0
best_acc = 0.0
best_kappa = 0.0
# Training
# ==================================================

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

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(),max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch1, y_batch2, y_batch3):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y1: y_batch1,
              cnn.input_y2: y_batch2,
              cnn.input_y3: y_batch3,
              cnn.dropout_keep_prob: config.dropout_keep_prob
            }
            _, step, loss, acc1, acc2, acc3 = sess.run(
               [train_op, global_step, cnn.loss, cnn.accuracy1, cnn.accuracy2, cnn.accuracy3],
               feed_dict)
            return step, loss, acc1, acc2, acc3

        def dev_step(x_batch, y_batch1, y_batch2, y_batch3):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y1: y_batch1,
                cnn.input_y2: y_batch2,
                cnn.input_y3: y_batch3,
                cnn.dropout_keep_prob: 1.0
            }
            loss, yhat1, yhat2, yhat3 = sess.run(
                   [cnn.loss, cnn.predictions1, cnn.predictions2, cnn.predictions3], feed_dict)
            return yhat1, yhat2, yhat3

        start_time = time.time()
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            train_loss_epoch = 0.0
            step = 1
            while step < train_batches_per_epoch:
                # Get a batch
                x_batch, y_batch1, label_batch1,\
                    y_batch2, label_batch2, \
                    y_batch3, label_batch3 = train_generator.next_batch(config.batch_size_per_class)

                x_batch = np.expand_dims(x_batch,axis=3)
                train_step_, train_loss_, train_acc1_, train_acc2_, train_acc3_ = train_step(x_batch, y_batch1, y_batch2, y_batch3)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {}, acc1 {}, acc2 {}, acc3 {}".format(time_str, train_step_, train_loss_,
                                                                               train_acc1_, train_acc2_, train_acc3_))
                train_loss_epoch += train_loss_
                step += 1

                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0:
                    # Validate the model on the entire evaluation test set after each epoch
                    print("{} Start validation".format(datetime.now()))

                    eval_yhat1 = np.zeros_like(eval_generator.data_index)
                    eval_yhat2 = np.zeros_like(eval_generator.data_index)
                    eval_yhat3 = np.zeros_like(eval_generator.data_index)
                    eval_step = 1
                    while eval_step < eval_batches_per_epoch:
                        x_batch, y_batch1, label_batch1_,\
                            y_batch2, label_batch2_,\
                            y_batch3, label_batch3_ = eval_generator.next_batch(config.batch_size)
                        x_batch = np.expand_dims(x_batch,axis=3)
                        eval_yhat1_, eval_yhat2_, eval_yhat3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
                        eval_yhat1[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat1_
                        eval_yhat2[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat2_
                        eval_yhat3[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat3_
                        eval_step += 1
                    if(eval_generator.pointer < len(eval_generator.data_index)):
                        actual_len, x_batch, y_batch1, label_batch1_, \
                            y_batch2, label_batch2_, \
                            y_batch3, label_batch3_ = eval_generator.rest_batch(config.batch_size)
                        x_batch = np.expand_dims(x_batch,axis=3)
                        eval_yhat1_, eval_yhat2_, eval_yhat3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
                        eval_yhat1[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat1_
                        eval_yhat2[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat2_
                        eval_yhat3[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat3_
                    eval_yhat1 = eval_yhat1 + 1
                    eval_yhat2 = eval_yhat2 + 1
                    eval_yhat3 = eval_yhat3 + 1
                    eval_acc1 = accuracy_score(eval_generator.label[:-1], eval_yhat1[1:])
                    eval_acc2 = accuracy_score(eval_generator.label, eval_yhat2)
                    eval_acc3 = accuracy_score(eval_generator.label[1:], eval_yhat3[:-1])
                    eval_fscore1 = f1_score(eval_generator.label[:-1], eval_yhat1[1:], average='macro')
                    eval_fscore2 = f1_score(eval_generator.label, eval_yhat2, average='macro')
                    eval_fscore3 = f1_score(eval_generator.label[1:], eval_yhat3[:-1], average='macro')
                    eval_kappa1 = cohen_kappa_score(eval_generator.label[:-1], eval_yhat1[1:])
                    eval_kappa2 = cohen_kappa_score(eval_generator.label, eval_yhat2)
                    eval_kappa3 = cohen_kappa_score(eval_generator.label[1:], eval_yhat3[:-1])


                    test_yhat1 = np.zeros_like(test_generator.data_index)
                    test_yhat2 = np.zeros_like(test_generator.data_index)
                    test_yhat3 = np.zeros_like(test_generator.data_index)
                    test_step = 1
                    while test_step < test_batches_per_epoch:
                        x_batch, y_batch1, label_batch1_,\
                            y_batch2, label_batch2_,\
                            y_batch3, label_batch3_ = test_generator.next_batch(config.batch_size)
                        x_batch = np.expand_dims(x_batch,axis=3)
                        test_yhat1_, test_yhat2_, test_yhat3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
                        test_yhat1[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat1_
                        test_yhat2[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat2_
                        test_yhat3[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat3_
                        test_step += 1
                    if(test_generator.pointer < len(test_generator.data_index)):
                        actual_len, x_batch, y_batch1, label_batch1_, \
                            y_batch2, label_batch2_, \
                            y_batch3, label_batch3_ = test_generator.rest_batch(config.batch_size)
                        x_batch = np.expand_dims(x_batch,axis=3)
                        test_yhat1_, test_yhat2_, test_yhat3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
                        test_yhat1[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat1_
                        test_yhat2[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat2_
                        test_yhat3[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat3_
                    test_yhat1 = test_yhat1 + 1
                    test_yhat2 = test_yhat2 + 1
                    test_yhat3 = test_yhat3 + 1
                    test_acc1 = accuracy_score(test_generator.label[:-1], test_yhat1[1:])
                    test_acc2 = accuracy_score(test_generator.label, test_yhat2)
                    test_acc3 = accuracy_score(test_generator.label[1:], test_yhat3[:-1])

                    print("{:g} {:g} {:g} {:g} {:g} {:g}".format(eval_acc1, eval_acc2,  eval_acc3, test_acc1, test_acc2,  test_acc3))
                    with open(os.path.join(out_dir, "result_log.txt"), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g} {:g}\n".format(eval_acc1, eval_acc2,  eval_acc3, test_acc1, test_acc2,  test_acc3))

                    # save the best model based on accuracy on evaluation data
                    if((eval_acc1 + eval_acc2 + eval_acc3)/3 > best_acc):

                        best_acc = (eval_acc1 + eval_acc2 + eval_acc3)/3 # update best performance

                        checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                        save_path = saver.save(sess, checkpoint_name)
                        print("Best model updated")
                        source_file = checkpoint_name
                        dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')

                    eval_generator.reset_pointer()
                    test_generator.reset_pointer()

            train_generator.reset_pointer()

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
