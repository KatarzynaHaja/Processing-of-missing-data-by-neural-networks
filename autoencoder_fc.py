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
from layer_builder import LayerBuilder
import analitic_method

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoencoderFCParams:
    def __init__(self, method, dataset, num_sample=None, num_layer_decoder=3, num_layer_encoder=3):
        self.num_hidden_1 = 256  # 1st layer num features
        self.num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_3 = 64  # 3nd layer num features (the latent dim)
        self.num_input = 784 if dataset == 'mnist' else 3072
        self.num_layer_decoder = num_layer_decoder
        self.num_layer_encoder = num_layer_encoder
        self.nn = 100
        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample
        self.n_distribution = 5  # number of n_distribution

        self.layer_inputs_encoder = [self.num_input, self.num_hidden_1, self.num_hidden_2]
        self.layer_outputs_encoder = [self.num_hidden_1, self.num_hidden_2, self.num_hidden_3]

        self.layer_inputs_decoder = [self.num_hidden_3, self.num_hidden_2, self.num_hidden_1]
        self.layer_outputs_decoder = [self.num_hidden_2, self.num_hidden_1, self.num_input]



class AutoencoderFC:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
        self.layer_builder = LayerBuilder(params)
        self.X = tf.placeholder("float", [None, self.params.num_input])
        self.original = tf.placeholder("float", [None, self.params.num_input])
        self.gamma = gamma
        self.gamma_int = gamma

        self.x_miss, self.x_known = self.divide_data_into_known_and_missing(self.X)
        self.size = tf.shape(self.x_miss)

        if self.params.method not in ['imputation', 'theirs']:
            self.sampling = Sampling(num_sample=self.params.num_sample, params=self.params, x_miss=self.x_miss,
                                     n_distribution=self.params.n_distribution,
                                     method=self.params.method)

    def set_variables(self):
        if self.params.method != 'imputation':
            gmm = GaussianMixture(n_components=self.params.n_distribution, covariance_type='diag').fit(
                self.data_imputed_train)
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
            layer_1 = self.layer_builder.build_layer(self.x_known, self.params.layer_inputs_encoder[0],
                                                     self.params.layer_outputs_encoder[0], 'relu')
            if self.params.method == 'theirs':
                layer_1 = analitic_method.nr(self.gamma, self.covs, self.p, self.params, self.means,
                                             self.x_miss, self.x_known, self.layer_builder.weights_1_layer,
                                             self.layer_builder.bias_1_layer)
            else:
                samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs,
                                                         self.params.num_input,
                                                         self.gamma)

                layer_1_miss = self.sampling.nr(samples, self.layer_builder.weights_1_layer,
                                                self.layer_builder.bias_1_layer)
                layer_1_miss = tf.reshape(layer_1_miss, shape=(self.size[0], self.params.num_hidden_1))

                layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)

        if self.params.method == 'imputation':
            layer_1 = self.layer_builder.build_layer(self.X, self.params.layer_inputs_encoder[0],
                                                     self.params.layer_outputs_encoder[0], 'relu')

        layer = layer_1

        for i in range(2):
            layer = self.layer_builder.build_layer(layer, self.params.layer_inputs_encoder[i + 1],
                                                   self.params.layer_outputs_encoder[i + 1], 'sigmoid')

        return layer

    def decoder(self, x):
        layer = x
        for i in range(3):
            layer = self.layer_builder.build_layer(layer, self.params.layer_inputs_decoder[i],
                                                   self.params.layer_outputs_decoder[i],
                                                   'sigmoid')

        if self.params.method != 'last_layer':
            return layer

        if self.params.method == 'last_layer':
            input = layer[:self.size[0] * self.params.num_sample, :]
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
            loss = tf.reduce_mean(tf.reduce_mean(tf.pow(y_true_miss - y_pred_miss, 2), axis=1),
                                  axis=0)  # srednia najpierw (weznatrz po obrazku a potem po samplu)

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        v = Visualizator(
            'result_' + str(self.params.method) + '_' + str(n_epochs) + '_' + str(self.params.num_sample) + '_' + str(
                self.gamma_int), 'loss', 100)

        train_loss = []
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
                train_loss.append(l)

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

        return np.mean(test_loss), test_loss, train_loss


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
            loss, test_loss, train_loss = a.autoencoder_main_loop(params['epoch'])
            print(method + "," + str(params['num_sample']) + ',' + str(params['epoch']) + ',' + str(
                params['gamma']) + ',' + str(loss))
            f.write(method + ", num_sample:" + str(params['num_sample']) + ', epoch:'
                    + str(params['epoch']) + ',' + str(params['gamma']) + 'result:' + str(loss))
            f.write('\n')
    f.close()


def run_the_best():
    dataset_processor = DatasetProcessor(dataset='mnist')
    X = dataset_processor.load_data()
    data_test, data_train = dataset_processor.divide_dataset_into_test_and_train(X)
    data_test = np.random.permutation(data_test)
    data_test, data_train = dataset_processor.change_background(data_test, data_train)
    # for j in range(1000):
    #     _, ax = plt.subplots(1, 1, figsize=(1, 1))
    #     ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
    #     ax.axis('off')
    #     plt.savefig(os.path.join('original_data', "".join(
    #         (str(j) + '.png'))),
    #                 bbox_inches='tight')
    #     plt.close()

    data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    # for j in range(1000):
    #     _, ax = plt.subplots(1, 1, figsize=(1, 1))
    #     ax.imshow(data_test[j].reshape([28, 28]), origin="upper", cmap="gray")
    #     ax.axis('off')
    #     plt.savefig(os.path.join('image_with_patch', "".join(
    #         (str(j) + '.png'))),
    #                 bbox_inches='tight')
    #     plt.close()

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
    train_losses = []
    test_losses = []
    for i in params:
        for params in i['params']:
            p = AutoencoderFCParams(method=i['method'], dataset='mnist', num_sample=params['num_sample'])
            a = AutoencoderFC(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                              data_imputed_test=data_imputed_test,
                              gamma=params['gamma'])
            loss, test_loss, train_loss = a.autoencoder_main_loop(params['epoch'])
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(i['method'] + "," + str(params['num_sample']) + ',' + str(params['epoch']) + ',' + str(
                params['gamma']) + ',' + str(loss))
            f.write(i['method'] + ", num_sample:" + str(params['num_sample']) + ', epoch:'
                    + str(params['epoch']) + ',' + str(params['gamma']) + 'result:' + str(loss))
            f.write('\n')
            f.write('Test loss:'+ ', '.join(test_loss))
            f.write('Train loss:' + ', '.join(train_loss))
            f.write('\n')
    f.close()

    Visualizator.draw_losses(train_losses)
    Visualizator.draw_losses(test_losses)


run_the_best()
# run_model('svhn')
