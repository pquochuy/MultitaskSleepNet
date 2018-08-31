# Test to see how good the DNN perform on the test data (just for double check)
# Extract and save the learn filterbank too. This will be used to preprocess the input data for other classification networks
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from scipy.io import loadmat,savemat
import h5py

from dnn_filterbank_config import Config
from dnn_filterbank import DNN_FilterBank

from datagenerator_from_list_v2 import DataGenerator

from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("test_data", "../data/test_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.8)")

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


config = Config()
config.dropout_keep_prob = FLAGS.dropout_keep_prob

test_generator = DataGenerator(os.path.abspath(FLAGS.test_data), shuffle = False)
test_batches_per_epoch = np.floor(test_generator.data_size / config.batch_size).astype(np.int16)


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        dnn = DNN_FilterBank(config=config)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(dnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model")
        saver.restore(sess, best_dir)
        print("Model loaded")

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

        Wfb, W, Wbl = sess.run([dnn.Wfb, dnn.W, dnn.Wbl])
        savemat(os.path.join(out_path, "filterbank.mat"), dict(Wfb=Wfb, W=W, Wbl=Wbl))

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
        test_yhat = test_yhat + 1
        test_fscore = f1_score(test_generator.label, test_yhat, average='macro')
        test_acc = accuracy_score(test_generator.label, test_yhat)
        test_kappa = cohen_kappa_score(test_generator.label, test_yhat)
        savemat(os.path.join(out_path, "test_ret.mat"), dict(acc=test_acc, fscore=test_fscore, kappa=test_kappa, yhat=test_yhat))
