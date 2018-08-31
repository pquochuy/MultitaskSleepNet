import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers

"""
Predefine all necessary layer for the R-CNN
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        #weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        #biases = tf.get_variable('biases', shape=[num_filters])

        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels, num_filters],
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)
        #print(bias.get_shape())
        #bias = tf.reshape(bias, conv.get_shape().as_list())
        bias = tf.reshape(bias, tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu

def linear_conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        #weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        #biases = tf.get_variable('biases', shape=[num_filters])
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels, num_filters],
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)
        #print(bias.get_shape())
        #bias = tf.reshape(bias, conv.get_shape().as_list())
        bias = tf.reshape(bias, tf.shape(conv))

        # Apply relu function
        #relu = tf.nn.relu(bias, name=scope.name)

        return bias


def fc(x, num_in, num_out, name, relu=True):
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights',
                                  shape=[num_in, num_out],
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        biases = tf.get_variable('biases',
                                 shape=[num_out],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)  # Apply ReLu non linearity
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

