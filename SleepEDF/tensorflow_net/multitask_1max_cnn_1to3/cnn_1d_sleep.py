import tensorflow as tf
import numpy as np


class CNN1DSleep(object):
    """
    A CNN for audio event classification.
    Uses a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, time_length, freq_length, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, time_length, freq_length,1], name="input_x")
        self.input_y1 = tf.placeholder(tf.float32, [None, num_classes], name="input_y1")
        self.input_y2 = tf.placeholder(tf.float32, [None, num_classes], name="input_y2")
        self.input_y3 = tf.placeholder(tf.float32, [None, num_classes], name="input_y3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # dim expansion
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/gpu:0'), tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, freq_length, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.input_x,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_max = tf.nn.max_pool(h,
                                            ksize=[1, time_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool")
                pooled_outputs.append(pooled_max)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.device('/gpu:0'), tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.device('/gpu:0'), tf.name_scope("output1"):
            W1 = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="scores1")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions1")

        # Final (unnormalized) scores and predictions
        with tf.device('/gpu:0'), tf.name_scope("output2"):
            W2 = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W2, b2, name="scores2")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions2")

        # Final (unnormalized) scores and predictions
        with tf.device('/gpu:0'), tf.name_scope("output3"):
            W3 = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b3")
            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)
            self.scores3 = tf.nn.xw_plus_b(self.h_drop, W3, b3, name="scores3")
            self.predictions3 = tf.argmax(self.scores3, 1, name="predictions3")

        # CalculateMean cross-entropy loss
        with tf.device('/gpu:0'), tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y1, logits=self.scores1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y2, logits=self.scores2)
            losses3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y3, logits=self.scores3)
            self.loss = tf.reduce_mean(losses1) + tf.reduce_mean(losses3) + tf.reduce_mean(losses2) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_y1, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")

            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_y2, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy2")

            correct_predictions3 = tf.equal(self.predictions3, tf.argmax(self.input_y3, 1))
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_predictions3, "float"), name="accuracy3")
