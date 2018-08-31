# import tensorflow as tf
import numpy as np
import os


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):

        self.n_dim = 129 # size of feature dimension

        # Trainging
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.0001
        self.training_epoch = 50 #50 epochs are sufficient as MASS is much larger than Sleep-EDF
        self.batch_size = 200
        self.batch_size_per_class = 40
        self.dropout_keep_prob = 0.8
        self.evaluate_every = 50


        self.n_class = 5

        self.n_layer = 3
        self.n_hidden_1 = 512  # nb of neurons inside the neural network
        self.n_hidden_2 = 256  # nb of neurons inside the neural network
        self.n_hidden_3 = 512  # nb of neurons inside the neural network

        # parameters to generate triangular filterbank shape matrix
        self.nfilter = 20
        self.f_min = 0
        self.f_max = 50
        self.f_high = 50
        self.f_low = 0
