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
        self.Y1 = tf.placeholder(tf.float32, shape=[None, self.config.n_class], name='Y1')
        self.Y2 = tf.placeholder(tf.float32, shape=[None, self.config.n_class], name='Y2')
        self.Y3 = tf.placeholder(tf.float32, shape=[None, self.config.n_class], name='Y3')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob_conv = tf.placeholder(tf.float32, name="dropout_keep_prob_conv")

        self.create()

    def create(self): # sorry for hardcoding parameters here
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
        flattened = tf.reshape(dropout2, [-1, 13 * 4 * 96])
        print(flattened.get_shape())
        fc3 = self.fc(flattened, 13 * 4 * 96, 1024, name='fc3')
        print(fc3.get_shape())
        dropout3 = self.dropout(fc3, self.dropout_keep_prob)

        # 4th Layer: FC (w ReLu) -> Dropout
        fc4 = self.fc(dropout3, 1024, 1024, name='fc4')
        print(fc4.get_shape())
        dropout4 = self.dropout(fc4, self.dropout_keep_prob)

        # 5th Layer: Output layer
        output_layer1 = self.fc(dropout4, 1024,
                               self.config.n_class,
                               name='output1',
                               relu=False)
        # 5th Layer: Output layer
        output_layer2 = self.fc(dropout4, 1024,
                               self.config.n_class,
                               name='output2',
                               relu=False)
        # 5th Layer: Output layer
        output_layer3 = self.fc(dropout4, 1024,
                               self.config.n_class,
                               name='output3',
                               relu=False)

        with tf.name_scope("prediction"):
            self.score1 = output_layer1
            self.score2 = output_layer2
            self.score3 = output_layer3
            self.pred_Y1 = tf.argmax(self.score1, 1, name='pred_Y1')
            self.pred_Y2 = tf.argmax(self.score2, 1, name='pred_Y2')
            self.pred_Y3 = tf.argmax(self.score3, 1, name='pred_Y3')

        with tf.name_scope("loss"):
            output_loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y1, logits=self.score1)
            output_loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y2, logits=self.score2)
            output_loss3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y3, logits=self.score3)
            output_loss = tf.reduce_mean(output_loss1) + tf.reduce_mean(output_loss2) + tf.reduce_mean(output_loss3)

        # add on regularization
        with tf.name_scope("l2_loss"):
            l2_loss = self.config.l2_reg_lambda * \
                      sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            self.loss = output_loss + l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.pred_Y1, tf.argmax(self.Y1, 1))
            correct_predictions2 = tf.equal(self.pred_Y1, tf.argmax(self.Y2, 1))
            correct_predictions3 = tf.equal(self.pred_Y1, tf.argmax(self.Y3, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy2")
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_predictions3, "float"), name="accuracy3")


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

