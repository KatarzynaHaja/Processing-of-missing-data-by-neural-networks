import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from sampling_cnn import Sampling
import numpy as np
import sys
import os

from visualization import Visualizator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoencoderCNNParams:
    def __init__(self, method, dataset, num_sample=None, learning_rate=0.001):
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.width = 28
        self.length = 28
        self.num_channels = 1

        self.encoder_layers = {
            'conv1_filters': [3, 3, 1, 16],
            'max_pool_1_ksize': 2,
            'max_pool_1_stride': 2,
            'conv2_filters': [3, 3, 16, 8],
            'max_pool_2_ksize': 2,
            'max_pool_2_stride': 2,
            'conv3_filters': [3, 3, 8, 8],
            'max_pool_3_ksize': 2,
            'max_pool_3_stride': 2,

        }

        self.decoder_layers = {
            'resize_1': (7, 7),
            'conv1_filters': [3, 3, 8, 8],
            'resize_2': (14, 14),
            'conv2_filters': [3, 3, 8, 8],
            'resize_3': (28, 28),
            'conv3_filters': [3, 3, 8, 16],
            'conv4_filters': [3, 3, 16, 1]

        }

        self.num_output = 10

        self.num_layers = 6

        self.nn = 100
        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample
        self.learning_rate = learning_rate

        self.encoder_filters = [self.encoder_layers['conv1_filters'], self.encoder_layers['conv2_filters'],
                                self.encoder_layers['conv3_filters']]
        self.decoder_filters = [self.decoder_layers['conv1_filters'], self.decoder_layers['conv2_filters'],
                                self.decoder_layers['conv3_filters'], self.decoder_layers['conv4_filters']]
        self.encoder_filters_weights = []
        self.encoder_filters_biases = []
        self.decoder_filters_weights = []
        self.decoder_filters_biases = []

        # initilize weights and biases for filters
        for i in range(len(self.encoder_filters)):
            self.encoder_filters_weights.append(tf.Variable(initializer(self.encoder_filters[i])))
            self.encoder_filters_biases.append(tf.Variable(tf.random_normal([self.encoder_filters[i][-1]])))

        for i in range(len(self.decoder_filters)):
            self.decoder_filters_weights.append(tf.Variable(initializer(self.decoder_filters[i])))
            self.decoder_filters_biases.append(tf.Variable(tf.random_normal([self.decoder_filters[i][-1]])))


class AutoencoderCNN:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
        self.n_distribution = 5
        self.X = tf.placeholder("float", [None, self.params.width * self.params.length])
        self.gamma = gamma
        self.gamma_int = gamma

        self.gamma = tf.Variable(initial_value=self.gamma)

        self.size = tf.shape(self.X)

        if self.params.method != 'imputation':
            self.x_miss, self.x_known = self.divide_data_into_known_and_missing(self.X)
            self.size = tf.shape(self.x_miss)
            self.sampling = Sampling(num_sample=self.params.num_sample, params=self.params, x_miss=self.x_miss,
                                     n_distribution=self.n_distribution,
                                     method=self.params.method)

        if self.params.method == 'imputation':
            self.reshaped_X = tf.reshape(self.X,
                                         shape=(
                                             self.size[0], self.params.width, self.params.length,
                                             self.params.num_channels))

    def set_variables(self):
        if self.params.method != 'imputation':
            gmm = GaussianMixture(n_components=self.n_distribution, covariance_type='diag').fit(
                self.data_imputed_train.reshape(self.data_imputed_train.shape[0], 784))
            self.p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
            self.means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
            self.covs = tf.abs(tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32))
            self.gamma = tf.Variable(initial_value=self.gamma)

    def divide_data_into_known_and_missing(self, x):
        check_isnan = tf.is_nan(x)
        check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

        x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))
        x_known = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))

        return x_miss, x_known

    def encoder(self):

        conv_1 = None

        if self.params.method != 'imputation':
            samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs,
                                                     self.params.width * self.params.length,
                                                     self.gamma)

            conv_1 = self.sampling.nr_autoencoder(samples)

        if self.params.method == 'imputation':
            conv_1 = tf.nn.relu(
                tf.add(tf.nn.conv2d(input=self.reshaped_X, filters=self.params.encoder_filters_weights[0], strides=1,
                                    padding='SAME'),
                       self.params.encoder_filters_biases[0]))

        # Now 28x28x16

        max_pooling_1 = tf.nn.max_pool2d(input=conv_1, ksize=self.params.encoder_layers['max_pool_1_ksize'],
                                         strides=self.params.encoder_layers['max_pool_1_stride'], padding='SAME')
        # Now 14x14x16
        conv_2 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=max_pooling_1, filters=self.params.encoder_filters_weights[1], strides=1,
                                padding='SAME'),
                   self.params.encoder_filters_biases[1]))

        # Now 14x14x8

        max_pooling_2 = tf.nn.max_pool2d(input=conv_2, ksize=self.params.encoder_layers['max_pool_2_ksize'],
                                         strides=self.params.encoder_layers['max_pool_2_stride'], padding='SAME')

        # Now 7x7x8
        conv_3 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=max_pooling_2, filters=self.params.encoder_filters_weights[2], strides=1,
                                padding='SAME'),
                   self.params.encoder_filters_biases[2]))
        # Now 7x7x8
        encoded = tf.nn.max_pool2d(input=conv_3, ksize=self.params.encoder_layers['max_pool_3_ksize'],
                                   strides=self.params.encoder_layers['max_pool_3_stride'], padding='SAME')

        # Now 4x4x8

        return encoded

    def decoder(self, encoded):
        upsample1 = tf.image.resize_nearest_neighbor(encoded, self.params.decoder_layers['resize_1'])
        # Now 7x7x8

        conv_4 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=upsample1, filters=self.params.decoder_filters_weights[0], strides=1,
                                padding='SAME'),
                   self.params.decoder_filters_biases[0]))

        # Now 7x7x8
        upsample2 = tf.image.resize_nearest_neighbor(conv_4, self.params.decoder_layers['resize_2'])
        # Now 14x14x8
        conv_5 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=upsample2, filters=self.params.decoder_filters_weights[1], strides=1,
                                padding='SAME'),
                   self.params.decoder_filters_biases[1]))

        # Now 14x14x8
        upsample3 = tf.image.resize_nearest_neighbor(conv_5, self.params.decoder_layers['resize_3'])
        # Now 28x28x8
        conv_6 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=upsample3, filters=self.params.decoder_filters_weights[2], strides=1,
                                padding='SAME'),
                   self.params.decoder_filters_biases[2]))

        # Now 28x28x16
        logits = tf.add(tf.nn.conv2d(input=conv_6, filters=self.params.decoder_filters_weights[3], strides=1,
                                     padding='SAME'),
                        self.params.decoder_filters_biases[3])

        if self.params.method != 'last_layer':
            return logits

        if self.params.method == 'last_layer':
            input = logits[:self.size[0] * self.params.num_sample, :]
            layer_fc_2 = self.sampling.mean_sample_autoencoder(input)
            return layer_fc_2

        return logits

    def autoencoder_main_loop(self, n_epochs):
        learning_rate = 0.01
        batch_size = 64

        loss = None

        self.set_variables()

        encoder_op = self.encoder()
        decoder_op = self.decoder(encoder_op)

        y_pred = decoder_op  # prediction
        y_true = tf.reshape(self.X, shape=(self.size[0], self.params.width, self.params.length,
                                           self.params.num_channels))  # z nanami

        if self.params.method != 'different_cost':
            where_isnan = tf.is_nan(y_true)
            y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
            y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)
            loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        if self.params.method == 'different_cost':
            y_true = tf.expand_dims(y_true, 0)
            y_true = tf.tile(y_true, [self.params.num_sample, 1, 1, 1, 1])
            y_true = tf.reshape(y_true, shape=(self.params.num_sample * self.size[0], self.params.width, self.params.width, self.params.num_channels))
            where_isnan = tf.is_nan(y_true)
            y_pred_miss = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)[
                          :self.size[0] * self.params.num_sample, :]
            y_true_miss = tf.where(where_isnan, tf.zeros_like(y_true), y_true)[
                          :self.size[0] * self.params.num_sample, :]
            loss_miss = tf.reduce_mean(tf.reduce_mean(tf.pow(y_true_miss - y_pred_miss, 2), axis=1),
                                       axis=0)  # srednia najpierw (weznatrz po obrazku a potem po samplu)
            # loss_known = tf.reduce_mean(tf.pow(y_true_known - y_pred_known, 2)) czy tutaj nie powinno być axis=1 zamiast axis=2 ???
            loss = loss_miss

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        v = Visualizator(
            'result_cnn_' + str(self.params.method) + '_' + str(n_epochs) + '_' + str(
                self.params.num_sample) + '_' + str(
                self.gamma_int), 'loss', 100)
        train_loss = []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, n_epochs + 1):
                losses = []
                n_batches = self.data_train.shape[0] // batch_size
                l = np.inf
                for iteration in range(n_batches):
                    print("\r{}% ".format(100 * (iteration + 1) // n_batches), end="")
                    sys.stdout.flush()

                    if self.params.method != 'imputation':
                        batch_x = self.data_train[(iteration * batch_size):((iteration + 1) * batch_size), :]
                    else:
                        batch_x = self.data_imputed_train[(iteration * batch_size):((iteration + 1) * batch_size), :]

                    _, l = sess.run([optimizer, loss], feed_dict={self.X: batch_x})
                    losses.append(l)
                print("Train loss", sum(losses) / len(losses))

                train_loss.append(sum(losses) / len(losses))

            test_loss = []
            if self.params.dataset == 'mnist':
                for i in range(10):
                    if self.params.method != 'imputation':
                        batch_x = self.data_test[(i * self.params.nn):((i + 1) * self.params.nn), :]
                    else:
                        batch_x = self.data_imputed_test[(i * self.params.nn):((i + 1) * self.params.nn), :]

                    g, l_test = sess.run([decoder_op, loss], feed_dict={self.X: batch_x})
                    for j in range(self.params.nn):
                        v.draw_mnist_image(i, j, g, self.params.method)
                    test_loss.append(l_test)

            if self.params.dataset == 'svhn':
                n = 16
                for i in range(n):
                    if self.params.method != 'imputation':
                        batch_x = self.data_test[
                                  (i * (self.data_test.shape[0]) // n):((i + 1) * (self.data_test.shape[0] // n)), :]
                    else:
                        batch_x = self.data_imputed_test[
                                  (i * (self.data_test.shape[0] // n)):((i + 1) * (self.data_test.shape[0] // n)), :]

                    g, l_test = sess.run([decoder_op, loss], feed_dict={self.X: batch_x})

                    for j in range(self.data_test.shape[0] // n):
                        v.draw_svhn_image(i, j, g, self.params.method)
                    test_loss.append(l_test)

        return train_loss, np.mean(test_loss)


def run_model():
    import matplotlib.pyplot as plt
    dataset_processor = DatasetProcessor(dataset='mnist')
    X = dataset_processor.load_data()
    data_test, data_train = dataset_processor.divide_dataset_into_test_and_train(X)
    data_test = np.random.permutation(data_test)
    data_test, data_train = dataset_processor.change_background(data_test, data_train)
    for j in range(1000):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join('original_data_cnn', "".join(
            (str(j) + '.png'))),
                    bbox_inches='tight')
        plt.close()

    data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    for j in range(1000):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join('image_with_patch_cnn', "".join(
            (str(j) + '.png'))),
                    bbox_inches='tight')
        plt.close()

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)

    data_imputed_train = imp.fit_transform(data_train)
    data_imputed_test = imp.transform(data_test)

    params = [
        {'method': 'imputation', 'params': [{'num_sample': 1, 'epoch': 50, 'gamma': 0.0}]},
        {'method': 'first_layer', 'params': [{'num_sample': 10, 'epoch': 50, 'gamma': 1.5},
                                             {'num_sample': 10, 'epoch': 50, 'gamma': 0.0},
                                             {'num_sample': 10, 'epoch': 50, 'gamma': 0.5}]},
        {'method': 'last_layer', 'params': [{'num_sample': 10, 'epoch': 50, 'gamma': 0.0},
                                            {'num_sample': 10, 'epoch': 50, 'gamma': 1.5},
                                            {'num_sample': 20, 'epoch': 50, 'gamma': 0.5},
                                            {'num_sample': 100, 'epoch': 50, 'gamma': 1.0},
                                            {'num_sample': 100, 'epoch': 50, 'gamma': 0.0}]},
        {'method': 'different_cost', 'params': [{'num_sample': 10, 'epoch': 50, 'gamma': 0.5},
                                                {'num_sample': 10, 'epoch': 50, 'gamma': 0.0},
                                                {'num_sample': 10, 'epoch': 50, 'gamma': 1.5}]}

    ]
    f = open('loss_results_autoencoder_conv', "a")
    for eleme in params:
        for param in eleme['params']:
            p = AutoencoderCNNParams(method=eleme['method'], dataset='mnist', num_sample=param['num_sample'])
            a = AutoencoderCNN(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                               data_imputed_test=data_imputed_test,
                               gamma=param['gamma'])
            train_loss, test_loss = a.autoencoder_main_loop(param['epoch'])
            f.write(eleme['method'] + "," + str(param['num_sample']) + ','
                    + str(param['epoch']) + ',' + str(param['gamma']) + ',' + str(test_loss))
            f.write('\n')
            f.write('Test loss:' + str(test_loss))
            f.write('\n')
            f.write('Train loss:' + str(train_loss))
            f.write('\n')
    f.close()


run_model()
