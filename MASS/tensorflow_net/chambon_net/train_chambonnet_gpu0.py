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

import time

from scipy.io import loadmat


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


eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eeg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.ntime], shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data), data_shape=[config.ntime], shuffle = False, test_mode=True)

    eeg_train_gen.X = np.expand_dims(eeg_train_gen.X, axis=-1) # expand feature dimension
    eeg_test_gen.X = np.expand_dims(eeg_test_gen.X, axis=-1) # expand feature dimension
    eeg_eval_gen.X = np.expand_dims(eeg_eval_gen.X, axis=-1) # expand feature dimension

if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.ntime], shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data), data_shape=[config.ntime], shuffle = False, test_mode=True)

    eog_train_gen.X = np.expand_dims(eog_train_gen.X, axis=-1) # expand feature dimension
    eog_test_gen.X = np.expand_dims(eog_test_gen.X, axis=-1) # expand feature dimension
    eog_eval_gen.X = np.expand_dims(eog_eval_gen.X, axis=-1) # expand feature dimension


if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = EqualDataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.ntime], shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.ntime], shuffle = False, test_mode=True)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data), data_shape=[config.ntime], shuffle = False, test_mode=True)

    emg_train_gen.X = np.expand_dims(emg_train_gen.X, axis=-1) # expand feature dimension
    emg_test_gen.X = np.expand_dims(emg_test_gen.X, axis=-1) # expand feature dimension
    emg_eval_gen.X = np.expand_dims(emg_eval_gen.X, axis=-1) # expand feature dimension

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen
eval_generator = eeg_eval_gen


if (not(eog_active) and not(emg_active)):
    train_generator.X = np.expand_dims(train_generator.X, axis=-1) # expand channel dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.expand_dims(test_generator.X, axis=-1) # expand channel dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.expand_dims(eval_generator.X, axis=-1) # expand channel dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 1
    print(train_generator.X.shape)

if (eog_active and not(emg_active)):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 2
    print(train_generator.X.shape)

if (eog_active and emg_active):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    print(emg_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X, emg_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X, emg_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X, emg_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 3
    print(train_generator.X.shape)

config.nchannel = nchannel

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

# shuffle training data here
train_generator.shuffle_data()

train_batches_per_epoch = np.floor((train_generator.data_size - len(train_generator.boundary_index)) / config.batch_size).astype(np.int16)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.int16)

print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(train_generator.data_size, eval_generator.data_size, test_generator.data_size))

print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

# variable to keep track of best fscore
min_valid_loss = float("inf")
# Training
# ==================================================

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())


        def train_step(x_batch, x_emg_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_x_emg: x_emg_batch,
              cnn.input_y: y_batch,
              cnn.dropout: config.dropout
            }
            _, step, loss, acc = sess.run(
               [train_op, global_step, cnn.loss, cnn.accuracy],
               feed_dict)
            return step, loss, acc

        def dev_step(x_batch, x_emg_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_x_emg: x_emg_batch,
                cnn.input_y: y_batch,
                cnn.dropout: 1.0
            }
            _, total_loss, output_loss, yhat, score, acc = sess.run(
                [global_step, cnn.loss, cnn.output_loss, cnn.predictions, cnn.score, cnn.accuracy],
                feed_dict)
            return output_loss, acc, yhat, score

        early_stop_count = 0
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            early_stop_count += 1

            train_loss_epoch = 0.0
            step = 1
            while step < train_batches_per_epoch:
                # Get a batch
                x_batch, y_batch, label_batch = train_generator.next_batch(config.batch_size_per_class)
                #(3*batchsize, T, 1, C)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                train_step_, train_loss_, train_acc_ = train_step(x_batch, x_emg_batch, y_batch)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {}, accuracy {}".format(time_str, train_step_, train_loss_, train_acc_))

                train_loss_epoch += train_loss_
                step += 1

            print("{} Start validation".format(datetime.now()))
            # evaluation
            test_yhat = np.zeros_like(test_generator.label)
            test_step = 1
            test_loss = 0.0
            while test_step < test_batches_per_epoch:
                x_batch, y_batch, _ = test_generator.next_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_, _, test_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat_
                test_step += 1
                test_loss += output_loss_
            if(test_generator.pointer < test_generator.data_size):
                actual_len, x_batch, y_batch, _ = test_generator.rest_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_,_, test_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                test_yhat[(test_step-1)*config.batch_size : test_generator.data_size] = test_yhat_
                test_loss += output_loss_
            test_fscore = f1_score(test_generator.label, test_yhat + 1, average='macro')
            test_acc = accuracy_score(test_generator.label, test_yhat + 1)
            test_kappa = cohen_kappa_score(test_generator.label, test_yhat + 1)


            eval_yhat = np.zeros_like(eval_generator.label)
            eval_step = 1
            valid_loss = 0.0
            while eval_step < eval_batches_per_epoch:
                x_batch, y_batch, _ = eval_generator.next_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_, _, eval_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                eval_yhat[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat_
                eval_step += 1
                valid_loss += output_loss_
            if(eval_generator.pointer < eval_generator.data_size):
                actual_len, x_batch, y_batch, _ = eval_generator.rest_batch(config.batch_size)

                x_emg_batch = x_batch[:,:,:,-1] # last channel is EMG
                x_emg_batch = np.expand_dims(x_emg_batch, axis=-1) # expand channel dimension
                x_batch = x_batch[:,:,:,0:-1] # last channel is EMG

                output_loss_, _, eval_yhat_, _ = dev_step(x_batch, x_emg_batch, y_batch)
                eval_yhat[(eval_step-1)*config.batch_size : eval_generator.data_size] = eval_yhat_
                valid_loss += output_loss_
            eval_fscore = f1_score(eval_generator.label, eval_yhat + 1, average='macro')
            eval_acc = accuracy_score(eval_generator.label, eval_yhat + 1)
            eval_kappa = cohen_kappa_score(eval_generator.label, eval_yhat + 1)



            print("{:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g}".format(valid_loss, eval_acc, eval_fscore, eval_kappa,
                                                                   test_loss, test_acc, test_fscore,  test_kappa))
            with open(os.path.join(out_dir, "result_log.txt"), "a") as text_file:
                text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g}\n".format(valid_loss, eval_acc, eval_fscore, eval_kappa,
                                                                         test_loss, test_acc, test_fscore,  test_kappa))

            if((valid_loss < min_valid_loss)):
                early_stop_count = 0
                min_valid_loss = valid_loss
                checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(epoch) +'.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                print("Best model updated")
                source_file = checkpoint_name
                dest_file = os.path.join(checkpoint_path, 'best_model')
                shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                shutil.copy(source_file + '.index', dest_file + '.index')
                shutil.copy(source_file + '.meta', dest_file + '.meta')

            if(early_stop_count >= 5):
                quit()

            # Reset the file pointer of the data generators
            test_generator.reset_pointer()
            eval_generator.reset_pointer()
            train_generator.reset_pointer()
