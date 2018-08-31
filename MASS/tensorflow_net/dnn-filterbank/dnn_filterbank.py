import tensorflow as tf
import numpy as np
import os
from dnn_filterbank_config import Config

class DNN_FilterBank(object):
    """
    A DNN for EEG filterbank learning
    """

    def __init__(self, config):
        self.config = config
        # place holder for feature vectors
        self.X = tf.placeholder(tf.float32, shape=[None, self.config.n_dim], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.n_class], name='y')
        # place holder for dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # triangular filterbank
        self.Wbl = tf.constant(self.tri_filter_shape(self.config.n_dim,
                                                     self.config.nfilter,
                                                     self.config.fmin,
                                                     self.config.fmax,
                                                     self.config.flow,
                                                     self.config.fhigh), dtype=tf.float32, name="W-filter-shape")

        # first filter bank layer
        with tf.device('/gpu:0'), tf.variable_scope("filterbank-layer-1"):
            self.W = tf.Variable(tf.random_normal([self.config.n_dim, self.config.nfilter],dtype=tf.float32))

            # non-negative constraints
            self.W = tf.sigmoid(self.W)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            self.Wfb = tf.multiply(self.W,self.Wbl)
            filtered_freq = tf.matmul(self.X, self.Wfb) # filtering

        self.HW = {
            'h1': tf.Variable(tf.random_normal([self.config.nfilter, self.config.n_hidden_1],stddev=0.1, dtype=tf.float32)),
            'h2': tf.Variable(tf.random_normal([self.config.n_hidden_1, self.config.n_hidden_2],stddev=0.1, dtype=tf.float32)),
            'h3': tf.Variable(tf.random_normal([self.config.n_hidden_2, self.config.n_hidden_3],stddev=0.1, dtype=tf.float32)),
            'output': tf.Variable(tf.random_normal([self.config.n_hidden_3, self.config.n_class],stddev=0.1, dtype=tf.float32))
        }
        self.biases = {
            'h1': tf.Variable(tf.random_normal([self.config.n_hidden_1], mean=1.0, dtype=tf.float32)),
            'h2': tf.Variable(tf.random_normal([self.config.n_hidden_2], mean=1.0, dtype=tf.float32)),
            'h3': tf.Variable(tf.random_normal([self.config.n_hidden_3], mean=1.0, dtype=tf.float32)),
            'output': tf.Variable(tf.random_normal([self.config.n_class], dtype=tf.float32))
        }


        with tf.device('/gpu:0'), tf.variable_scope("fully-connected-layers"):
            H1 = tf.add(tf.matmul(filtered_freq, self.HW['h1']), self.biases['h1'])
            H1 = tf.nn.relu(H1)
            dropout_layer1 = tf.nn.dropout(H1, self.dropout_keep_prob)
            H2 = tf.add(tf.matmul(dropout_layer1, self.HW['h2']), self.biases['h2'])
            H2 = tf.nn.relu(H2)
            dropout_layer2 = tf.nn.dropout(H2, self.dropout_keep_prob)
            H3 = tf.add(tf.matmul(dropout_layer2, self.HW['h3']), self.biases['h3'])
            H3 = tf.nn.relu(H3)
            dropout_layer3 = tf.nn.dropout(H3, self.dropout_keep_prob)

        with tf.device('/gpu:0'), tf.variable_scope("output"):
            output = tf.add(tf.matmul(dropout_layer3, self.HW['output']), self.biases['output'])
            self.score = output # logit
            self.pred_Y = tf.argmax(self.score, 1, name='pred_Y') # predicted labels

        # calculate cross-entropy loss
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.score)
            self.output_loss = tf.reduce_mean(self.output_loss)

        # add on regularization (optional)
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            #l2_loss = self.config.l2_reg_lambda * \
            #          sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            #self.loss = self.output_loss + l2_loss

            # here to exclude l2-norm constraint on the filterbank weights
            #l2_loss += tf.nn.l2_loss(self.HW['h1'])
            #l2_loss += tf.nn.l2_loss(self.HW['h2'])
            #l2_loss += tf.nn.l2_loss(self.HW['h3'])
            #l2_loss += tf.nn.l2_loss(self.HW['output'])
            #l2_loss += tf.nn.l2_loss(self.biases['h1'])
            #l2_loss += tf.nn.l2_loss(self.biases['h2'])
            #l2_loss += tf.nn.l2_loss(self.biases['h3'])
            #l2_loss += tf.nn.l2_loss(self.biases['output'])
            #self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss
            self.loss = self.output_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.pred_Y, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def dropout_layer(self, input_layer, keep_prob):
        return tf.nn.dropout(input_layer, keep_prob)

    def fully_connected_layer(self, input_layer, input_size, output_size):
        W = tf.Variable(tf.random_normal([input_size, output_size],stddev=0.1, dtype=tf.float32))
        bias = tf.Variable(tf.random_normal([output_size], stddev=0.1, dtype=tf.float32))
        return tf.add(tf.matmul(input_layer, W), bias)

    def tri_filter_shape(self, ndim, nfilter, fmin, fmax, flow, fhigh):
        f_min = fmin
        f_max = fmax
        f_high = fhigh
        f_low = flow
        f = np.linspace(f_min, f_max, ndim)

        H = np.zeros((nfilter, ndim))

        M = f_low + np.arange(nfilter+2)*(f_high-f_low)/(nfilter+1)
        for m in range(nfilter):
            k = np.logical_and(f >= M[m], f <= M[m+1])   # up-slope
            H[m][k] = 2*(f[k]-M[m]) / ((M[m+2]-M[m])*(M[m+1]-M[m]))
            k = np.logical_and(f >= M[m+1], f <= M[m+2]) # down-slope
            H[m][k] = 2*(M[m+2] - f[k]) / ((M[m+2]-M[m])*(M[m+2]-M[m+1]))

        H = np.transpose(H)
        H.astype(np.float32)
        return H

