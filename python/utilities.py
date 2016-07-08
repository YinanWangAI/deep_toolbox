"""Common used functions"""
import tensorflow as tf


def binomial_sample(prob):
    """Sample from a series of Bernoulli distributions

    :param prob: numpy array, p(x=1) for each node
    :return: tensorflow object, the sampled data
    """

    sample_out = tf.select(tf.random_uniform(shape=prob.get_shape(), minval=0, maxval=1) < prob,
                           t=tf.ones(prob.get_shape()), e=tf.zeros(prob.get_shape()))
    return sample_out
