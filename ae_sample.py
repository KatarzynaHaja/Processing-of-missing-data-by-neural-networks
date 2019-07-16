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


def random_one_sample(p, x_miss):
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)
    component = random_component(p)
    data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[component, :], [size[0]]), [-1, size[1]]), x_miss)
    miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[component, :], [size[0]]), [-1, size[1]]),
                        tf.zeros([size[0], size[1]]))
    return random_from_component(data_miss, miss_cov)


def random_samples(num_samples, p, x_miss):
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)
    samples = random_one_sample(p, x_miss)
    for i in range(num_samples-1):
        component = random_component(p)
        data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[component, :], [size[0]]), [-1, size[1]]), x_miss)
        miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[component, :], [size[0]]), [-1, size[1]]),
                            tf.zeros([size[0], size[1]]))
        samples = tf.concat(samples, random_from_component(data_miss, miss_cov))

    return samples


def return_layer(output, size):
    reshaped_output = tf.reshape(output, shape=(size[0] * size[1], num_input))
    layer_1_m = tf.add(tf.matmul(reshaped_output, weights['encoder_h1']), biases['encoder_b1'])
    layer_1_m = tf.nn.relu(layer_1_m)
    unreshaped = tf.reshape(layer_1_m, shape=(size[0], size[1], num_input))
    mean = tf.reduce_mean(unreshaped, 0)
    return mean




# Building the encoder
def encoder(x, means, covs, p, gamma):
    gamma = tf.abs(gamma)
    gamma_ = tf.cond(tf.less(gamma[0], 1.), lambda: gamma, lambda: tf.pow(gamma, 2))
    covs = tf.abs(covs)
    p = tf.abs(p)
    p = tf.div(p, tf.reduce_sum(p, axis=0))
    p_ = tf.nn.softmax(p)

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values

    # data without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    # data  with missing

    weights2 = tf.square(weights['encoder_h1'])
    size = tf.shape(x_miss)
    output = random_one_sample(p_,x_miss)
    layer_1_miss = return_layer(output, size)
    print(layer_1_miss.get_shape())
    print(layer_1.get_shape())
    #
    # # join layer for data_rbfn with missing values with layer for data_rbfn without missing values
    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)
    #
    # # Encoder Hidden layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3


def prep_x(x):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((x, x_miss), axis=0)


t0 = time()
mnist = input_data.read_data_sets("./data_mnist/", one_hot=True)
print("Read data_rbfn done in %0.3fs." % (time() - t0))

data_train = mnist.train.images

# choose test images nn * 10
nn = 100
labels = np.where(mnist.test.labels == 1)[1]
data_test = mnist.test.images[np.where(labels == 0)[0][:nn], :]
for i in range(1, 10):
    data_test = np.concatenate([data_test, mnist.test.images[np.where(labels == i)[0][:nn], :]], axis=0)
data_test = np.random.permutation(data_test)

del mnist, labels

# change background to white
data_train = 1. - data_train
data_test = 1. - data_test

# create missing data_rbfn train
data_train = data_with_mask(data_train, width_mask)

# create missing data_rbfn test
data_test = data_with_mask(data_test, width_mask)

imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
data = imp.fit_transform(data_train)

t0 = time()
gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(data)
print("GMM done in %0.3fs." % (time() - t0))

p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
covs = tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32)
gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)
del data, gmm

# Construct model
encoder_op = encoder(X, means, covs, p, gamma)
decoder_op = decoder(encoder_op)

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

    for epoch in range(1, n_epochs + 1):
        n_batches = data_train.shape[0] // batch_size
        l = np.inf
        for iteration in range(n_batches):
            print("\r{}% ".format(100 * (iteration + 1) // n_batches), end="")
            sys.stdout.flush()

            batch_x = data_train[(iteration * batch_size):((iteration + 1) * batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display loss per step
        print('Step {:d}: Minibatch Loss: {:.8f}, gamma: {:.4f}'.format(epoch, l, gamma.eval()[0]))

    # results for test data_rbfn
    for i in range(10):
        batch_x = data_test[(i * nn):((i + 1) * nn), :]

        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display reconstructed images
        for j in range(nn):
            # Draw the reconstructed digits
            _, ax = plt.subplots(1, 1, figsize=(1, 1))
            ax.imshow(g[j].reshape([28, 28]), origin="upper", cmap="gray")
            ax.axis('off')
            plt.savefig(os.path.join(save_dir, "".join(
                (str(i * nn + j), "-sample.png"))),
                        bbox_inches='tight')
            plt.close()