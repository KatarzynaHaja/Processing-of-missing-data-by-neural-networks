import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from ae_sample import Sampling
from visualization import Visualizator
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoencoderFCParams:
    def __init__(self, method, dataset, num_sample=None, ):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.num_hidden_1 = 256  # 1st layer num features
        self.num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_3 = 64  # 3nd layer num features (the latent dim)
        self.num_input = 784 if dataset == 'mnist' else 3072

        self.nn = 100
        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample

        self.weights = {
            'encoder_h1': tf.Variable(initializer([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(initializer([self.num_hidden_1, self.num_hidden_2])),
            'encoder_h3': tf.Variable(initializer([self.num_hidden_2, self.num_hidden_3])),
            'decoder_h1': tf.Variable(initializer([self.num_hidden_3, self.num_hidden_2])),
            'decoder_h2': tf.Variable(initializer([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h3': tf.Variable(initializer([self.num_hidden_1, self.num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.num_input])),
        }


class AutoencoderFC:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
        self.n_distribution = 5  # number of n_distribution
        self.X = tf.placeholder("float", [None, self.params.num_input])
        self.original = tf.placeholder("float", [None, self.params.num_input])
        self.gamma = gamma
        self.gamma_int = gamma

        self.x_miss, self.x_known = self.divide_data_into_known_and_missing(self.X)
        self.size = tf.shape(self.x_miss)

        if self.params.method != 'imputation':
            self.sampling = Sampling(num_sample=self.params.num_sample, params=self.params, x_miss=self.x_miss,
                                     n_distribution=self.n_distribution,
                                     method=self.params.method)

    def set_variables(self):
        if self.params.method != 'imputation':
            gmm = GaussianMixture(n_components=self.n_distribution, covariance_type='diag').fit(self.data_imputed_train)
            self.p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
            self.means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
            self.covs = tf.abs(tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32))
            if self.params.method != 'theirs':
                self.gamma = tf.Variable(initial_value=self.gamma)
            else:
                self.gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.),
                                         dtype=tf.float32)

    def divide_data_into_known_and_missing(self, x):
        check_isnan = tf.is_nan(x)
        check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

        x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
        x_known = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values

        return x_miss, x_known

    def encoder(self):
        if self.params.method != 'imputation':
            size = tf.shape(self.x_miss)

            if self.params.method == 'theirs':
                gamma = tf.abs(self.gamma)
                gamma_ = tf.cond(tf.less(gamma[0], 1.), lambda: gamma, lambda: tf.pow(gamma, 2))
                covs = tf.abs(self.covs)
                p = tf.abs(self.p)
                p = tf.div(p, tf.reduce_sum(p, axis=0))

                layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x_known, self.params.weights['encoder_h1']),
                                            self.params.biases['encoder_b1']))

                where_isnan = tf.is_nan(self.x_miss)
                where_isfinite = tf.is_finite(self.x_miss)

                weights2 = tf.square(self.params.weights['encoder_h1'])

                Q = []
                layer_1_miss = tf.zeros([size[0], self.params.num_hidden_1])
                for i in range(self.n_distribution):
                    data_miss = tf.where(where_isnan, tf.reshape(tf.tile(self.means[i, :], [size[0]]), [-1, size[1]]),
                                         self.x_miss)
                    miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[i, :], [size[0]]), [-1, size[1]]),
                                        tf.zeros([size[0], size[1]]))

                    layer_1_m = tf.add(tf.matmul(data_miss, self.params.weights['encoder_h1']),
                                       self.params.biases['encoder_b1'])

                    layer_1_m = tf.div(layer_1_m, tf.sqrt(tf.matmul(miss_cov, weights2)))
                    layer_1_m = tf.div(tf.exp(tf.div(-tf.pow(layer_1_m, 2), 2.)), np.sqrt(2 * np.pi)) + tf.multiply(
                        tf.div(layer_1_m, 2.), 1 + tf.erf(
                            tf.div(layer_1_m, np.sqrt(2))))

                    layer_1_miss = tf.cond(tf.equal(tf.constant(i), tf.constant(0)),
                                           lambda: tf.add(layer_1_miss, layer_1_m),
                                           lambda: tf.concat((layer_1_miss, layer_1_m), axis=0))

                    norm = tf.subtract(data_miss, self.means[i, :])
                    norm = tf.square(norm)
                    q = tf.where(where_isfinite,
                                 tf.reshape(tf.tile(tf.add(gamma_, covs[i, :]), [size[0]]), [-1, size[1]]),
                                 tf.ones_like(self.x_miss))
                    norm = tf.div(norm, q)
                    norm = tf.reduce_sum(norm, axis=1)

                    q = tf.log(q)
                    q = tf.reduce_sum(q, axis=1)

                    q = tf.add(q, norm)

                    norm = tf.cast(tf.reduce_sum(tf.cast(where_isfinite, tf.int32), axis=1), tf.float32)
                    norm = tf.multiply(norm, tf.log(2 * np.pi))

                    q = tf.add(q, norm)
                    q = tf.multiply(q, -0.5)

                    Q = tf.concat((Q, q), axis=0)

                Q = tf.reshape(Q, shape=(self.n_distribution, -1))
                Q = tf.add(Q, tf.log(p))
                Q = tf.subtract(Q, tf.reduce_max(Q, axis=0))
                Q = tf.where(Q < -20, tf.multiply(tf.ones_like(Q), -20), Q)
                Q = tf.exp(Q)
                Q = tf.div(Q, tf.reduce_sum(Q, axis=0))
                Q = tf.reshape(Q, shape=(-1, 1))

                layer_1_miss = tf.multiply(layer_1_miss, Q)
                layer_1_miss = tf.reshape(layer_1_miss, shape=(self.n_distribution, size[0], self.params.num_hidden_1))
                layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)
                layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)

            else:

                samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs,
                                                         self.params.num_input,
                                                         self.gamma)
                layer_1 = tf.nn.relu(
                    tf.add(tf.matmul(self.x_known, self.params.weights['encoder_h1']),
                           self.params.biases['encoder_b1']))

                layer_1_miss = self.sampling.nr(samples)
                layer_1_miss = tf.reshape(layer_1_miss, shape=(size[0], self.params.num_hidden_1))

                layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)

        if self.params.method == 'imputation':
            layer_1 = tf.nn.relu(
                tf.add(tf.matmul(self.X, self.params.weights['encoder_h1']),
                       self.params.biases['encoder_b1']))

        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_1, self.params.weights['encoder_h2']), self.params.biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.params.weights['encoder_h3']), self.params.biases['encoder_b3']))
        return layer_3

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(x, self.params.weights['decoder_h1']), self.params.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_1, self.params.weights['decoder_h2']), self.params.biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.params.weights['decoder_h3']), self.params.biases['decoder_b3']))

        if self.params.method != 'last_layer':
            return layer_3

        if self.params.method == 'last_layer':
            input = layer_3[:self.size[0] * self.params.num_sample, :]
            mean = self.sampling.mean_sample(input, self.params.num_input)
            return mean

    def autoencoder_main_loop(self, n_epochs):
        learning_rate = 0.01
        batch_size = 64

        loss = None

        self.set_variables()

        encoder_op = self.encoder()
        decoder_op = self.decoder(encoder_op)

        y_pred = decoder_op  # prediction
        y_true = tf.concat((self.x_known, self.x_miss), axis=0)  # z nanami

        if self.params.method != 'different_cost':
            where_isnan = tf.is_nan(y_true)
            y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
            y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)
            loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        if self.params.method == 'different_cost':
            y_true = tf.expand_dims(y_true, 0)
            y_true = tf.tile(y_true, [self.params.num_sample, 1, 1])
            y_true = tf.reshape(y_true, shape=(self.params.num_sample * self.size[0], self.params.num_input))
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
            'result_' + str(self.params.method) + '_' + str(n_epochs) + '_' + str(self.params.num_sample) + '_' + str(
                self.gamma_int), 'loss', 100)

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, n_epochs + 1):
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

        return np.mean(test_loss)


from sklearn.model_selection import ParameterGrid


def run_model(dataset):
    if dataset == 'mnist':
        dataset_processor = DatasetProcessor(dataset='mnist')
        X = dataset_processor.load_data()
        data_test, data_train = dataset_processor.divide_dataset_into_test_and_train(X)
        data_test = np.random.permutation(data_test)
        data_test, data_train = dataset_processor.change_background(data_test, data_train)
        data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    if dataset == 'svhn':
        d = DatasetProcessor('svhn')
        data_train, data_test, labels_train, labels_test = d.load_data()
        data_train = d.reshape_data(data_train)
        data_test = d.reshape_data(data_test)
        data_train, data_test = d.mask_data(data_train, data_test)

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data_imputed_train = imp.fit_transform(data_train)
    data_imputed_test = imp.transform(data_test)

    hyper_params = {'num_sample': [10], 'epoch': [150], 'gamma': [1.0]}
    methods = ['imputation']
    grid = ParameterGrid(hyper_params)
    f = open('loss_results_4', "a")
    for method in methods:
        for params in grid:
            print(method, params)
            p = AutoencoderFCParams(method=method, dataset='svhn', num_sample=params['num_sample'])
            a = AutoencoderFC(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                              data_imputed_test=data_imputed_test,
                              gamma=params['gamma'])
            loss = a.autoencoder_main_loop(params['epoch'])
            print(method + "," + str(params['num_sample']) + ',' + str(params['epoch']) + ',' + str(
                params['gamma']) + ',' + str(loss))
            f.write(method + ", num_sample:" + str(params['num_sample']) + ', epoch:'
                    + str(params['epoch']) + ',' + str(params['gamma']) + 'result:' + str(loss))
            f.write('\n')
    f.close()


def run_the_best():
    dataset_processor = DatasetProcessor(path='', dataset='mnist', width_mask=13, nn=100)
    X = dataset_processor.load_data()
    data_test, data_train = dataset_processor.divide_dataset_into_test_and_train(X)
    data_test = np.random.permutation(data_test)
    data_test, data_train = dataset_processor.change_background(data_test, data_train)
    for j in range(1000):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join('original_data', "".join(
            (str(j) + '.png'))),
                    bbox_inches='tight')
        plt.close()

    data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    for j in range(1000):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join('image_with_patch', "".join(
            (str(j) + '.png'))),
                    bbox_inches='tight')
        plt.close()

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data_imputed_train = imp.fit_transform(data_train)
    data_imputed_test = imp.transform(data_test)

    params = [{'method': 'theirs', 'params': [{'num_sample': 1, 'epoch': 250, 'gamma': 0.0}]},
              {'method': 'last_layer', 'params': [{'num_sample': 10, 'epoch': 150, 'gamma': 0.0},
                                                  {'num_sample': 20, 'epoch': 150, 'gamma': 0.0},
                                                  {'num_sample': 10, 'epoch': 150, 'gamma': 0.0},
                                                  {'num_sample': 20, 'epoch': 150, 'gamma': 1.0}]},
              {'method': 'first_layer', 'params': [{'num_sample': 10, 'epoch': 150, 'gamma': 1.0},
                                                   {'num_sample': 20, 'epoch': 150, 'gamma': 1.0}]},
              {'method': 'different_cost', 'params': [{'num_sample': 10, 'epoch': 250, 'gamma': 2.0}]},

              {'method': 'imputation', 'params': [{'num_sample': 1, 'epoch': 250, 'gamma': 0.0}]}

              ]
    f = open('loss_results_the_best', "a")
    for i in params:
        for params in i['params']:
            p = AutoencoderFCParams(method=i['method'], dataset='mnist', num_sample=params['num_sample'])
            a = AutoencoderFC(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                              data_imputed_test=data_imputed_test,
                              gamma=params['gamma'])
            loss = a.autoencoder_main_loop(params['epoch'])
            print(i['method'] + "," + str(params['num_sample']) + ',' + str(params['epoch']) + ',' + str(
                params['gamma']) + ',' + str(loss))
            f.write(i['method'] + ", num_sample:" + str(params['num_sample']) + ', epoch:'
                    + str(params['epoch']) + ',' + str(params['gamma']) + 'result:' + str(loss))
            f.write('\n')
    f.close()


# run_the_best()
run_model('svhn')
