# Using frame-wise data to train the DNN to learn frequency-domain filterbank

import os
# Explicitly indicate using gpu 0 to prevent tensorflow block all available gpus
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from dnn_filterbank_config import Config
from dnn_filterbank import DNN_FilterBank

from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

from datagenerator_from_list_v2 import DataGenerator
from equaldatagenerator_from_list_v2 import EqualDataGenerator

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

# Initalize the data generator seperately for the training, validation, and test sets
train_generator = EqualDataGenerator(os.path.abspath(FLAGS.train_data), shuffle = True)
test_generator = DataGenerator(os.path.abspath(FLAGS.test_data), shuffle = False)

# no normalization here for this network


config = Config()
config.dropout_keep_prob = FLAGS.dropout_keep_prob

train_batches_per_epoch = np.floor(train_generator.data_size / config.batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(test_generator.data_size / config.batch_size).astype(np.int16)

print("Train/Test set: {:d}/{:d}".format(train_generator.data_size, test_generator.data_size))

# variable to keep track of best fscore

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        dnn = DNN_FilterBank(config=config)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #optimizer = tf.train.AdamOptimizer(1e-4)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(dnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables())

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                dnn.X: x_batch,
                dnn.y: y_batch,
                dnn.dropout_keep_prob: config.dropout_keep_prob
            }
            _, step, loss, acc = sess.run(
               [train_op, global_step, dnn.loss, dnn.accuracy],
               feed_dict)
            return step, loss, acc

        def dev_step(x_batch, y_batch):
            feed_dict = {
                dnn.X: x_batch,
                dnn.y: y_batch,
                dnn.dropout_keep_prob: 1.0
            }
            _, loss, yhat, score, acc = sess.run(
                [global_step, dnn.loss, dnn.pred_Y, dnn.score, dnn.accuracy],
                feed_dict)
            return acc, yhat, score

        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            step = 1
            while step < train_batches_per_epoch:
                # Get a batch
                x_batch, y_batch, label_batch = train_generator.next_batch(config.batch_size_per_class)
                train_step_, train_loss_, train_acc_ = train_step(x_batch, y_batch)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {}, accuracy {}".format(time_str, train_step_, train_loss_, train_acc_))
                step += 1

            # Validate the model on the entire test set after each epoch to see how good it is
            # But we should not use the evaluation results for picking up the best model since the test data is unseen
            print("{} Start validation".format(datetime.now()))

            test_yhat = np.zeros_like(test_generator.label)
            test_step = 1
            while test_step < test_batches_per_epoch:
                x_batch, y_batch, _ = test_generator.next_batch(config.batch_size)
                _, test_yhat_, _ = dev_step(x_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat_
                test_step += 1
            if(test_generator.pointer < test_generator.data_size):
                actual_len, x_batch, y_batch, _ = test_generator.rest_batch(config.batch_size)
                _, test_yhat_, _ = dev_step(x_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : test_generator.data_size] = test_yhat_
            test_fscore = f1_score(test_generator.label, test_yhat + 1, average='macro')
            test_acc = accuracy_score(test_generator.label, test_yhat + 1)
            test_kappa = cohen_kappa_score(test_generator.label, test_yhat + 1)

            print("{:g} {:g} {:g}".format(test_acc, test_fscore,  test_kappa))
            with open(os.path.join(out_dir, "result_log.txt"), "a") as text_file:
                text_file.write("{:g} {:g} {:g}\n".format(test_acc, test_fscore,  test_kappa))

            # Reset the file pointer of the data generators
            train_generator.reset_pointer()
            test_generator.reset_pointer()

        # save the latest model here
        checkpoint_name = os.path.join(checkpoint_path, 'best_model')
        save_path = saver.save(sess, checkpoint_name)
