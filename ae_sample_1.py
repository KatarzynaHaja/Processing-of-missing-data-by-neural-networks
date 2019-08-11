import os
import pathlib
import sys
from time import time

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'








# def encoder(x, means, covs, p):
#     covs = tf.abs(covs)
#     p = tf.abs(p)
#     p_ = tf.nn.softmax(p)
#
#     check_isnan = tf.is_nan(x)
#     check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)
#
#     x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
#     x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values
#
#     # data without missing
#     layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
#
#     size = tf.shape(x_miss)
#     print("Size", size)
#     samples = create_data(num_sample, p_, x_miss, means, covs)
#     layer_1_miss = nr(samples, size, num_sample)
#
#     print("Layer 1 miss", layer_1_miss.get_shape())
#
#     layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)
#
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
#     return layer_3






    return data, data_test, data_train, gmm




def main():
    data, data_test, data_train, gmm = prepare_data()
    p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
    means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
    covs = tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32)


    # Construct model
    encoder_op = encoder(X, means, covs, p)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op  # prediction
    y_true = prep_x(X)  # Targets (Labels) are the input data_rbfn.

    where_isnan = tf.is_nan(y_true)
    y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
    y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    print(loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)  # run the initializer
        loss_k = []
        for epoch in range(1, n_epochs + 1):
            n_batches = data_train.shape[0] // batch_size
            l = np.inf
            for iteration in range(n_batches):
                print("\r{}% ".format(100 * (iteration + 1) // n_batches), end="")
                sys.stdout.flush()

                batch_x = data_train[(iteration * batch_size):((iteration + 1) * batch_size), :]

                _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

            print('Step {:d}: Minibatch Loss: {:.8f}'.format(epoch, l))
            loss_k.append(l)

        print(sum(loss_k) / 250)
        plot_loss(loss_k, [i for i in range(n_epochs)])

        for i in range(10):
            batch_x = data_test[(i * nn):((i + 1) * nn), :]

            g = sess.run(decoder_op, feed_dict={X: batch_x})

            # Display reconstructed images
            for j in range(nn):
                draw_image(i,j, g)
