import tensorflow as tf


class LayerBuilder:
    def __init__(self, params):
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.weights_1_layer = tf.Variable(self.initializer([params.num_input, params.num_hidden_1]))
        self.bias_1_layer = tf.Variable(tf.random_normal([params.num_hidden_1]))

    def build_layer(self, previous_layer, input, output, activation):
        weight = tf.Variable(self.initializer([input, output]))
        bias = tf.Variable(tf.random_normal([output]))
        if activation == 'sigmoid':
            return tf.nn.sigmoid(tf.add(tf.matmul(previous_layer, weight), bias))
        if activation == 'relu':
            return tf.nn.relu(tf.add(tf.matmul(previous_layer, weight), bias))
