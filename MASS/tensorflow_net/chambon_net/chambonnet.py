import tensorflow as tf

from nn_basic_layers import *

import numpy as np
import os
from config import Config

class ChambonNet(object):
    """
    A CNN for sleep staging.
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        self.input_x = tf.placeholder(tf.float32,shape=[None, self.config.ntime, self.config.ndim, self.config.nchannel-1],name='input_x')
        # EMG channel is handled separately (cf. Chambon et al.)
        self.input_x_emg = tf.placeholder(tf.float32,shape=[None, self.config.ntime, self.config.ndim, 1],name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.nclass], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name="dropout")


        X = tf.transpose(self.input_x, [0, 3, 1, 2]) #(3*batchsize, C, T, 1)
        print(X.get_shape()) #(3*batchsize, 1, T, C)

        # brach 1 (EEG and EOG)
        conv11 = linear_conv(X, self.config.nchannel-1, 1, self.config.nchannel-1, 1, 1, padding='VALID', name='conv11')
        print(conv11.get_shape()) #(3*batchsize, 1, T, C)
        conv11 = tf.transpose(conv11, [0, 3, 2, 1]) #(3*batchsize, C, T, 1)

        conv12 = conv(conv11, 1, 50, 8, 1, 1, padding='SAME',name='conv12') # 50 = Fs/2
        print(conv12.get_shape()) #(3*batchsize, C, T, 8) #(-1, 2, 3000, 8)
        # pooling size of 14 / stride 14 to get a decimation of approximately twice sampling rate as in Chambon et al.
        pool12 = max_pool(conv12, 1, 14, 1, 14, padding='SAME', name='pool12')
        print(pool12.get_shape()) #(-1, 2, 215, 8)

        conv13 = conv(pool12, 1, 50, 8, 1, 1, padding='SAME',name='conv13') # 50 = Fs/2
        print(conv13.get_shape()) #(3*batchsize, C, T/14, 8) #(-1, 2, 215, 8)
        # pooling size of 14 / stride 14 to get a decimation of approximately twice sampling rate as in Chambon et al.
        pool13 = max_pool(conv13, 1, 14, 1, 14,padding='SAME', name='pool13')
        print(pool13.get_shape()) #(-1, 2, 16, 8)

        cnn_output = tf.reshape(pool13, [-1, int((self.config.nchannel-1)*np.ceil(np.ceil(self.config.ntime/14)/14)*8)]) #[3*batchsize, 2*30*8]
        cnn_output = dropout(cnn_output, self.dropout)
        print(cnn_output.get_shape()) #(-1, 2, 16, 8)

        # concatenate features of 3 epochs in the contextual input
        feat0 = cnn_output[0::3] #[batchsize, 2*16*8]
        feat1 = cnn_output[1::3] #[batchsize, 2*16*8]
        feat2 = cnn_output[2::3] #[batchsize, 2*16*8]
        print(feat0.get_shape())
        print(feat1.get_shape())
        print(feat2.get_shape())
        feat = tf.concat([feat0, feat1, feat2], axis=1)
        print(feat.get_shape()) #(-1, 2, 16, 8)



        X_emg = tf.transpose(self.input_x_emg, [0, 3, 1, 2]) #(3*batchsize, C, T, 1)
        print(X_emg.get_shape()) #(3*batchsize, 1, T, C)

        # branch 2 (EMG branch)
        conv21 = linear_conv(X_emg, 1, 1, 1, 1, 1, padding='VALID', name='conv21')
        print(conv21.get_shape()) #(3*batchsize, 1, T, C)
        conv21 = tf.transpose(conv21, [0, 3, 2, 1]) #(3*batchsize, C, T, 1)

        conv22 = conv(conv21, 1, 50, 8, 1, 1, padding='SAME',name='conv22') # 50 = Fs/2
        print(conv22.get_shape()) #(3*batchsize, C, T, 8) #(-1, 2, 3000, 8)
        # pooling size of 14 / stride 14 to get a decimation of approximately twice sampling rate as in Chambon et al.
        pool22 = max_pool(conv22, 1, 14, 1, 14, padding='SAME', name='pool22')
        print(pool22.get_shape()) #(-1, 2, 215, 8)

        conv23 = conv(pool22, 1, 50, 8, 1, 1, padding='SAME',name='conv23') # 50 = Fs/2
        print(conv23.get_shape()) #(3*batchsize, C, T/14, 8) #(-1, 2, 215, 8)
        # pooling size of 14 / stride 14 to get a decimation of approximately twice sampling rate as in Chambon et al.
        pool23 = max_pool(conv23, 1, 14, 1, 14,padding='SAME', name='pool23')
        print(pool23.get_shape()) #(-1, 2, 16, 8)

        cnn_output_emg = tf.reshape(pool23, [-1, int(np.ceil(np.ceil(self.config.ntime/14)/14)*8)]) #[3*batchsize, 2*30*8]
        cnn_output_emg = dropout(cnn_output_emg, self.dropout)
        print(cnn_output_emg.get_shape()) #(-1, 2, 16, 8)

        # concatenate features of 3 epochs in the contextual input
        feat0_emg = cnn_output_emg[0::3] #[batchsize, 2*16*8]
        feat1_emg = cnn_output_emg[1::3] #[batchsize, 2*16*8]
        feat2_emg = cnn_output_emg[2::3] #[batchsize, 2*16*8]
        print(feat0_emg.get_shape())
        print(feat1_emg.get_shape())
        print(feat2_emg.get_shape())
        feat_emg = tf.concat([feat0_emg, feat1_emg, feat2_emg], axis=1)
        print(feat_emg.get_shape()) #(-1, 2, 16, 8)

        # concatenation two branches
        feat = tf.concat([feat, feat_emg], axis=1)


        nfeat = 3*self.config.nchannel*np.ceil(np.ceil(self.config.ntime/14)/14)*8
        print(nfeat)
        with tf.device('/gpu:0'), tf.variable_scope("output_layer"):
            self.score = fc(feat, nfeat, self.config.nclass, name="fc",relu=False)
            self.predictions = tf.argmax(self.score, 1, name="predictions")

        # cross-entropy output loss
        self.output_loss = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            self.output_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y), logits=self.score)
            self.output_loss = tf.reduce_sum(self.output_loss, axis=[0])

        # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.predictions, tf.argmax(tf.squeeze(self.input_y), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
