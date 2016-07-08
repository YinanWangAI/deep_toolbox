"""Deep Feedforward Networks, powered by TensorFlow"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class DeepFeedForwardNetworks(object):
    def __init__(self, feature_num, response_num, architecture, active_fun,
                 output="softmax"):
        self.x = self._create_placeholder([None, feature_num])
        self.y = self._create_placeholder([None, response_num])
        self.keep_prob = self._create_placeholder(None)  # for scaler, shape is None
        n_hidden_layers = len(architecture)
        for i in range(n_hidden_layers):
            if i == 0:
                new_input = self.x
                W_shape = [feature_num, architecture[i]]
            else:
                W_shape = [architecture[i - 1], architecture[i]]
            b_shape = [architecture[i]]
            hl = self._add_full_connected_layer(new_input, W_shape, b_shape)
            new_input = hl.activate(active_fun=active_fun[i])
            new_input = tf.nn.dropout(new_input, keep_prob=self.keep_prob)             # TODO: dropout still have bugs, the accuracy drops after some training iterations
        W_out = self._create_weights(shape=[architecture[-1], response_num])
        b_out = self._create_bias(shape=[response_num])
        ol = OutputLayer(new_input, W_out, b_out)
        if output == "linear":
            self.y_prediction = ol.linear_output()
        elif output == "sigmoid":
            self.y_prediction = ol.sigmoid_output()
        elif output == "softmax":
            self.y_prediction = ol.softmax_output()
        else:
            print("Wrong output function!")

    def train_model(self, x, y, batch_size, learning_rate=0.1,
                    optimizer=tf.train.GradientDescentOptimizer,
                    cost_fun="cross entropy", keep_probability=1.0,
                    training_step=10000,
                    test_x=None, test_y=None):
        if cost_fun == "cross entropy":
            cost = self._cross_entropy()
        elif cost_fun == "mean square":
            cost = self._mean_square()
        else:
            print("Wrong cost function")
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] >= batch_size
        batch_num = int(x.shape[0] / batch_size)

        train_step = optimizer(learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            run_num = 0
            correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1),
                                          tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            while run_num < training_step:
                for i in range(batch_num):
                    batch_x = x[i * batch_size: (i + 1) * batch_size]
                    batch_y = y[i * batch_size: (i + 1) * batch_size]
                    train_step.run(feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.keep_prob: keep_probability})
                    run_num += 1
                    if run_num % 100 == 0:
                        train_accuracy = accuracy.eval(feed_dict={
                            self.x: batch_x,
                            self.y: batch_y,
                            self.keep_prob: 1.0})
                        print("step %d, training accuracy %g" %
                              (run_num, train_accuracy))
            saver = tf.train.Saver()
            save_path = saver.save(sess, "/tmp/dfn.ckpt")
            print("Model saved in file: ", save_path)
            if (test_x is not None) and (test_y is not None):
                test_accuracy = accuracy.eval(feed_dict={
                    self.x: test_x, self.y: test_y, self.keep_prob: 1.0
                })
                print("Testing accuracy is %g" % test_accuracy)

    def predict(self, x, response_type="classification"):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "/tmp/dfn.ckpt")
            if response_type == "classification":
                y = tf.arg_max(self.y_prediction, 1)
            elif response_type == "regression":
                y = self.y_prediction
            else:
                print("Wrong response_type")
            return y.eval(feed_dict={self.x: x, self.keep_prob: 1.0})

    def _create_placeholder(self, shape, dtype=tf.float32):
        return tf.placeholder(dtype=dtype, shape=shape)

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1))

    def _create_bias(self, shape):
        return tf.Variable(tf.zeros(shape=shape))

    def _add_full_connected_layer(self, x, W_shape, b_shape):
        W = self._create_weights(W_shape)
        b = self._create_bias(b_shape)
        return HiddenLayer(x, W, b)

    def _mean_square(self):
        return tf.reduce_mean(tf.square(self.y - self.y_prediction))

    def _cross_entropy(self):
        return tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_prediction),
                                             reduction_indices=[1]))


class HiddenLayer(object):
    def __init__(self, x, W, b):
        """
        Initiate the HiddenLayer object
        :param x: TensorFlow placeholder, the input of hidden layer
        :param num_nodes: int, the number of nodes of the layer
        """
        self.x = x
        self.W = W
        self.b = b

    def activate(self, active_fun=tf.nn.relu):
        return active_fun(tf.matmul(self.x, self.W) + self.b)


class OutputLayer(object):
    """Output layer for a network"""

    def __init__(self, x, W, bias):
        self.x = x
        self.W = W
        self.bias = bias

    def linear_output(self):
        return tf.matmul(self.x, self.W) + self.bias

    def sigmoid_output(self):
        return tf.nn.sigmoid(self.linear_output())

    def softmax_output(self):
        return tf.nn.softmax(self.linear_output())


def test_DeepFeedForwardNetworks():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    classifier = DeepFeedForwardNetworks(784, 10,
                                         [800, 200],
                                         [tf.nn.relu, tf.nn.relu],
                                         output="softmax")
    classifier.train_model(train_x, train_y, batch_size=50,
                           test_x=test_x, test_y=test_y, training_step=5000,
                           keep_probability=0.8)
    print(classifier.predict(test_x))


if __name__ == "__main__":
    test_DeepFeedForwardNetworks()
