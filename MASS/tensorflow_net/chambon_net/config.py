# import tensorflow as tf
import numpy as np
import os


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # Trainging

        self.learning_rate = 1e-3
        self.l2_reg_lambda = 0.0 # no l2-norm regulariation
        self.training_epoch = 200
        self.batch_size = 100
        self.batch_size_per_class = 20 # if equal sampling
        # dropout keep probability
        self.dropout = 0.5


        self.ndim = 1  # feature dimension (i.e. raw signal)
        self.ntime = 3000  # time dimension
        self.nchannel = 3  # channel dimension

        self.nclass = 5  # Final output classes

