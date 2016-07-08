"""Restricted Boltzmann Machines

A class of Restriced Boltzmann Machines implemented by TensorFlow
"""

import numpy as np
import tensorflow as tf
from utilities import binomial_sample


class RBM(object):
    """Class for Restricted Boltzmann Machines"""

    def __init__(self, num_visible, num_hidden, batch_size):
        """Initiate TensorFlow variables and placeholders

        :param num_visible: int, the number of visible nodes
        :param num_hidden: int, the number of hidden nodes
        :param batch_size: int, the size for each batch
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = tf.Variable(tf.truncated_normal([num_visible, num_hidden]))
        self.v_bias = tf.Variable(tf.zeros([num_visible]))
        self.h_bias = tf.Variable(tf.zeros([num_hidden]))
        self.input_data = tf.placeholder(tf.float32,
                                         shape=[batch_size, num_visible],
                                         name='input_data')
        self.batch_size = batch_size

    def train_model(self, training_data,
                    learning_rate=0.1, k=1, max_training_step=100):
        """Train the RBM model

        :param training_data: numpy array, the data to train the model
        :param learning_rate: int, learning rate for gradient decent algorithm
        :param k: int, the number of sampling for cd
        :param max_training_step: int,
            the max number of iterations during training
        :return: numpy array, the cost after each training step, can be used to
            monitor the training process
        """

        # TODO: the support of sgd
        # TODO: the support of persistent cd algorithm

        [p_h_given_v0, h0, p_vk_given_h, vk] = self.gibbs_sampling_v(
            self.input_data, k=k)
        [p_h_given_vk, hk_sample] = self.sample_h_given_v(vk)
        W_grad = (-tf.matmul(tf.transpose(self.input_data), p_h_given_v0) +
                  tf.matmul(tf.transpose(vk), p_h_given_vk)) / self.batch_size
        v_bias_grad = tf.reduce_mean(vk - self.input_data, 0)
        h_bias_grad = tf.reduce_mean(p_h_given_vk - p_h_given_v0, 0)

        track_cost = []
        with tf.Session() as sess:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            for i in range(max_training_step):
                print(i)
                W_updates = self.W - learning_rate * W_grad
                v_bias_updates = self.v_bias - learning_rate * v_bias_grad
                h_bias_updates = self.h_bias - learning_rate * h_bias_grad
                assign_op = [self.W.assign(W_updates),
                             self.v_bias.assign(v_bias_updates),
                             self.h_bias.assign(h_bias_updates)]
                sess.run(assign_op, feed_dict={self.input_data: training_data})
                this_cost = self._get_pseudo_likelihood(training_data)
                track_cost.append(this_cost)
            # save the trained parameters
            self.W = self.W.eval()
            self.v_bias = self.v_bias.eval()
            self.h_bias = self.h_bias.eval()
        return track_cost

    def p_h_given_v(self, v):
        """Compute p(h=1|v)

        :param v: TensorFlow placeholder, visible data
        :return: TensorFlow placeholder, p(h=1|v)
        """

        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.h_bias)

    def p_v_given_h(self, h):
        """Compute p(v=1|h)

        :param h: TensorFlow placeholder, hidden data
        :return: TensorFlow placeholder, p(v=1|h)
        """

        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.v_bias)

    def sample_h_given_v(self, v0):
        """Sample hidden data given visible data

        :param v0: TensorFlow placeholder, visible data
        :return: TensorFlow placeholder, sampled hidden data
        """

        p_h0_given_v0 = self.p_h_given_v(v0)
        h0_sample = binomial_sample(p_h0_given_v0)
        return [p_h0_given_v0, h0_sample]

    def sample_v_given_h(self, h0):
        """Sample visible data given hidden data

        :param h0: TensorFlow placeholder, hidden data
        :return: TensorFlow placeholder, sampled visible data
        """

        p_v0_given_h0 = self.p_v_given_h(h0)
        v0_sample = binomial_sample(p_v0_given_h0)
        return [p_v0_given_h0, v0_sample]

    def gibbs_sampling_v(self, v0, k=1):
        """K-step gibbs sampling, sample visible data

        :param v0: TensorFlow placeholder, visible data
        :param k: int, the number of steps to run cd algorithm
        :return: list of TensorFlow placeholders,
          [p(h0=1|v0), h0, p(vk=1|h), vk]
        """

        sample_num = 0
        vk = v0
        while sample_num < k:
            [p_h_prev_given_v, h_prev] = self.sample_h_given_v(vk)
            if k == 1:
                [p_h0_given_v0, h0] = [p_h_prev_given_v, h_prev]
            [p_vk_given_h, vk] = self.sample_v_given_h(h_prev)
            sample_num += 1
        return [p_h0_given_v0, h0, p_vk_given_h, vk]

    def gibbs_sampling_h(self, h0, k=1):
        """K-step gibbs sampling, sample hidden data

        :param h0: TensorFlow placeholder, hidden data
        :param k: int, the number of steps to run cd algorithm
        :return: list of TensorFlow placeholders,
          [p(hk=1|v0), hk]
        """

        # TODO: make the return similar to gibbs_sampling_v
        sample_num = 0
        hk = h0
        while sample_num < k:
            [p_v0_given_h, v0] = self.sample_v_given_h(hk)
            [p_hk_given_v, hk] = self.sample_h_given_v(v0)
            sample_num += 1
        return [p_hk_given_v, hk]

    def _get_pseudo_likelihood(self, v0):
        """The difference between sampled data and real data

        :param v0: numpy array, real data
        :return: int, difference between sampled data and real data
        """

        v1 = self.gibbs_sampling_v(self.input_data, k=1)[3]
        v1_val = v1.eval(feed_dict={self.input_data: v0})
        return np.sum(np.abs(v1_val - v0))
