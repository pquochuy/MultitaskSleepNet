import tensorflow as tf
import numpy as np
import os
from cnn_sleep_config import Config

class CNN_Sleep(object):
    """
    A CNN for sleep staging.
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        self.X = tf.placeholder(tf.float32,
                                shape=[None, self.config.n_time, self.config.n_dim, self.config.n_channel],
                                name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.config.n_class], name='Y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob_conv = tf.placeholder(tf.float32, name="dropout_keep_prob_conv")

        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) --> Pool (time x frequency)
        conv1 = self.conv(self.X, 3, 3, 96, 1, 1, padding='VALID', name='conv1')
        print(conv1.get_shape())
        pool1 = self.max_pool(conv1, 1, 2, 1, 2,padding='VALID', name='pool1')
        print(pool1.get_shape())
        dropout1 = self.dropout(pool1, self.dropout_keep_prob_conv)

        # 2nd Layer: Conv (w ReLu) --> Pool (time x frequency)
        conv2 = self.conv(dropout1, 3, 3, 96, 1, 1, padding='VALID', name='conv2')
        print(conv2.get_shape())
        pool2 = self.max_pool(conv2, 2, 2, 2, 2, padding='SAME', name='pool2')
        print (pool2.get_shape())
        dropout2 = self.dropout(pool2, self.dropout_keep_prob_conv)

        # 3th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(dropout2, [-1, 42 * 4 * 96])
        print(flattened.get_shape())
        fc3 = self.fc(flattened, 42 * 4 * 96, 1024, name='fc3')
        print(fc3.get_shape())
        dropout3 = self.dropout(fc3, self.dropout_keep_prob)

        # 4th Layer: FC (w ReLu) -> Dropout
        fc4 = self.fc(dropout3, 1024, 1024, name='fc4')
        print(fc4.get_shape())
        dropout4 = self.dropout(fc4, self.dropout_keep_prob)

        # 5th Layer: Output layer
        output_layer = self.fc(dropout4, 1024,
                               self.config.n_class,
                               name='output',
                               relu=False)

        with tf.name_scope("prediction"):
            self.score = output_layer
            self.pred_Y = tf.argmax(self.score, 1, name='pred_Y')

        with tf.name_scope("loss"):
            output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.score)
            output_loss = tf.reduce_mean(output_loss)

        # add on regularization
        with tf.name_scope("l2_loss"):
            l2_loss = self.config.l2_reg_lambda * \
                      sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss = output_loss + l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.pred_Y, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    """
    Predefine necessary layers
    """

    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            conv = convolve(x, weights)

            # Add biases
            bias = tf.nn.bias_add(conv, biases)
            bias = tf.reshape(bias, tf.shape(conv))

            # Apply relu function
            relu = tf.nn.relu(bias, name=scope.name)
            return relu


    def fc(self, x, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', [num_out])

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            if relu == True:
                relu = tf.nn.relu(act)  # Apply ReLu non linearity
                return relu
            else:
                return act

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(x,
                              ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding,
                              name=name)

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
