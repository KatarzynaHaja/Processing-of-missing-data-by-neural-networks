import os
import pathlib
import sys
from time import time

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow_probability as tfp

from matplotlib import pyplot as plt

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# Training Parameters
learning_rate = 0.01
n_epochs = 250
batch_size = 64

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_hidden_3 = 64  # 3nd layer num features (the latent dim)
num_input = 784  # MNIST data_rbfn input (img shape: 28*28)

n_distribution = 5  # number of n_distribution
num_sample = 10
nn = 100

save_dir = "./results_mnist_sample/"
width_mask = 13  # size of window mask

pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(save_dir, 'images_png')).mkdir(parents=True, exist_ok=True)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])


initializer = tf.contrib.layers.variance_scaling_initializer()

weights = {
    'encoder_h1': tf.Variable(initializer([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(initializer([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(initializer([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(initializer([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(initializer([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(initializer([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}


def random_mask(width_window, margin=0):
    margin_left = margin
    margin_righ = margin
    margin_top = margin
    margin_bottom = margin
    start_width = margin_top + np.random.randint(28 - width_window - margin_top - margin_bottom)
    start_height = margin_left + np.random.randint(28 - width_window - margin_left - margin_righ)

    return np.concatenate([28 * i + np.arange(start_height, start_height + width_window) for i in
                           np.arange(start_width, start_width + width_window)], axis=0).astype(np.int32)


def data_with_mask(x, width_window=10):
    h = width_window
    for i in range(x.shape[0]):
        if width_window <= 0:
            h = np.random.randint(8, 20)
        maska = random_mask(h)
        x[i, maska] = np.nan
    return x


def random_component(p):
    return tf.squeeze(tf.gather(tf.random.multinomial(p, 1), 0))


def random_from_component(mu, sigma):
    mvn = tfd.MultivariateNormalDiag(
        loc=mu,
        scale_diag=sigma)
    return mvn.sample(1)


def get_distibution_params(component, x_miss, means, covs):
    where_isnan = tf.is_nan(x_miss)
    size = tf.shape(x_miss)
    data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[component, :], [size[0]]), [-1, size[1]]), x_miss)
    miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[component, :], [size[0]]), [-1, size[1]]),
                        tf.zeros([size[0], size[1]]))

    return data_miss, miss_cov


def create_data(num_sample, p, x_miss, means, covs):
    size = tf.shape(x_miss)
    samples = tf.zeros([1, size[0], num_input])

    for sam in range(num_sample):
        component = random_component(p)
        mu, sigma = get_distibution_params(component, x_miss, means, covs)
        sample = random_from_component(mu, sigma)
        samples = tf.cond(tf.equal(tf.constant(sam), tf.constant(0)), lambda: tf.add(samples, sample),
                               lambda: tf.concat((samples, sample), axis=0))

    print("samples", samples.get_shape())
    return samples


def nr(output, size, num_sample):
    print("Output", output.get_shape())
    reshaped_output = tf.reshape(output, shape=(size[0] * num_sample, num_input))
    layer_1_m = tf.add(tf.matmul(reshaped_output, weights['encoder_h1']), biases['encoder_b1'])
    layer_1_m = tf.nn.relu(layer_1_m)
    return layer_1_m


def mean_sample(input,size,  num_sample):
    unreshaped = tf.reshape(input, shape=(num_sample, size[0], 784))
    mean = tf.reduce_mean(unreshaped, axis=0)
    return mean


def plot_loss(loss, epochs):
    plt.plot(epochs, loss)
    plt.title('Loss - wylosowano 10 probek, uśrednianie na końcu sieci')
    plt.savefig(os.path.join(save_dir, "loss_1_sample.png"))
    plt.close()


def encoder(x, means, covs, p):
    covs = tf.abs(covs)
    p = tf.abs(p)
    p_ = tf.nn.softmax(p)

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values

    # data without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    size = tf.shape(x_miss)
    print("Size", size)
    samples = create_data(num_sample, p_, x_miss, means, covs)
    layer_1_miss = nr(samples, size, num_sample)
    layer_1_miss = tf.reshape(layer_1_miss, shape=(size[0], num_hidden_1))

    print("Layer 1 miss", layer_1_miss.get_shape())

    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3, size


# Building the decoder
def decoder(x, size):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    input = layer_3[:size[0] * num_sample, :]
    mean = mean_sample(input, size, num_sample)
    return mean


def prep_x(x):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((x, x_miss), axis=0)


def prepare_data():
    mnist = input_data.read_data_sets("./data_mnist/", one_hot=True)
    data_train = mnist.train.images
    labels = np.where(mnist.test.labels == 1)[1]
    data_test = mnist.test.images[np.where(labels == 0)[0][:nn], :]
    for i in range(1, 10):
        data_test = np.concatenate([data_test, mnist.test.images[np.where(labels == i)[0][:nn], :]], axis=0)
    data_test = np.random.permutation(data_test)
    data_train = 1. - data_train
    data_test = 1. - data_test

    data_train = data_with_mask(data_train, width_mask)

    data_test = data_with_mask(data_test, width_mask)

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imp.fit_transform(data_train)

    gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(data)

    return data, data_test, data_train, gmm


def draw_image(i, j,  g):
    _, ax = plt.subplots(1, 1, figsize=(1, 1))
    ax.imshow(g[j].reshape([28, 28]), origin="upper", cmap="gray")
    ax.axis('off')
    plt.savefig(os.path.join(save_dir, "".join(
        (str(i * nn + j), "-sample.png"))),
                bbox_inches='tight')
    plt.close()


def main():
    data, data_test, data_train, gmm = prepare_data()
    p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
    means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
    covs = tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32)


    # Construct model
    encoder_op, size = encoder(X, means, covs, p)
    decoder_op = decoder(encoder_op, size)

    y_pred = decoder_op  # prediction
    y_true = prep_x(X)  # Targets (Labels) are the input data_rbfn.

    where_isnan = tf.is_nan(y_true)
    y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
    y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
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
                draw_image(i, j, g)


main()
