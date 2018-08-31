# import tensorflow as tf
import numpy as np
import os


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):

        self.time_length = 29  # the number of time steps per series
        self.freq_length = 129

        # Trainging
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 200
        self.batch_size = 200
        self.batch_size_per_class = 40 # if equal sampling
        self.dropout_keep_prob = 0.8

        self.evaluate_every = 50

        self.filter_sizes = "3,5,7"  # sizes of filterbanks
        self.num_filters = 500 # number of conv. filters for each size
        self.num_classes = 5  # Number of output classes
