import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from chambonnet import ChambonNet
from config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator
from equaldatagenerator_from_list_v2 import EqualDataGenerator

from scipy.io import loadmat, savemat


# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability (default: 0.75)")

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
config.dropout = FLAGS.dropout

eeg_active = (FLAGS.eeg_test_data != "")
eog_active = (FLAGS.eog_test_data != "")
emg_active = (FLAGS.emg_test_data != "")

if (eeg_active):
    print("eeg active")
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    eeg_test_gen.X = np.expand_dims(eeg_test_gen.X, axis=-1) # expand feature dimension

if (eog_active):
    print("eog active")
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    eog_test_gen.X = np.expand_dims(eog_test_gen.X, axis=-1) # expand feature dimension


if (emg_active):
    print("emg active")
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    emg_test_gen.X = np.expand_dims(emg_test_gen.X, axis=-1) # expand feature dimension


# eeg always active
test_generator = eeg_test_gen

if (not(eog_active) and not(emg_active)):
    test_generator.X = np.expand_dims(test_generator.X, axis=-1) # expand channel dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 1

if (eog_active and not(emg_active)):
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 2

if (eog_active and emg_active):
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X, emg_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 3

config.nchannel = nchannel

del eeg_test_gen
if (eog_active):
    del eog_test_gen
if (emg_active):
    del emg_test_gen

# shuffle training data here
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)

print("Test set: {:d}".format(test_generator.data_size))

print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ChambonNet(config=config)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables())
        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model")
        saver.restore(sess, best_dir)
        print("Model loaded")


        def dev_step(x_batch, x_emg_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_x_emg: x_emg_batch,
                cnn.input_y: y_batch,
                cnn.dropout: 1.0
            }
            output_loss, total_loss, yhat, score = sess.run(
                   [cnn.output_loss, cnn.loss, cnn.predictions, cnn.score], feed_dict)
            return output_loss, total_loss, yhat, score

        def evaluate(gen):
            # Validate the model on the entire evaluation test set after each epoch
            total_loss = 0

            num_batch_per_epoch = np.floor(len(gen.data_index) / (config.batch_size)).astype(np.uint32)

            test_yhat = np.zeros_like(gen.label)
            test_step = 1
            test_loss = 0.0
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, _ = gen.next_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_, _, test_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat_
                test_step += 1
                test_loss += output_loss_
            if(gen.pointer < gen.data_size):
                actual_len, x_batch, y_batch, _ = gen.rest_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_,_, test_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : gen.data_size] = test_yhat_
                test_loss += output_loss_
            test_yhat += 1
            test_acc = accuracy_score(gen.label, test_yhat)

            return test_acc, test_yhat, total_loss


        test_acc, test_yhat, test_total_loss = evaluate(gen=test_generator)
        savemat(os.path.join(out_path, "test_ret.mat"), dict(yhat = test_yhat, acc = test_acc, total_loss = test_total_loss))
        test_generator.reset_pointer()

