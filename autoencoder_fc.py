import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from ae_sample import Sampling
from visualization import Visualizator
import numpy as np
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoencoderFCParams:
    def __init__(self, method, dataset, num_sample=None, ):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.num_hidden_1 = 256  # 1st layer num features
        self.num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_3 = 64  # 3nd layer num features (the latent dim)
        self.num_input = 784  # MNIST data_rbfn input (img shape: 28*28)

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
            self.gamma = tf.Variable(initial_value=self.gamma)

    def divide_data_into_known_and_missing(self, x):
        check_isnan = tf.is_nan(x)
        check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

        x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
        x_known = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values

        return x_miss, x_known

    def encoder(self):
        if self.params.method != 'imputation':
            size = tf.shape(self.x_miss)
            samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs, self.params.num_input,
                                                     self.gamma)
            layer_1 = tf.nn.relu(
                tf.add(tf.matmul(self.x_known, self.params.weights['encoder_h1']), self.params.biases['encoder_b1']))

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
            mean = self.sampling.mean_sample(input)
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

            y_pred_known = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)[
                           self.size[0] * self.params.num_sample:, :]
            y_true_known = tf.where(where_isnan, tf.zeros_like(y_true), y_true)[
                           self.size[0] * self.params.num_sample:, :]
            loss_miss = tf.reduce_mean(tf.reduce_mean(tf.pow(y_true_miss - y_pred_miss, 2), axis=1),
                                       axis=0)  # srednia najpierw (weznatrz po obrazku a potem po samplu)
            # loss_known = tf.reduce_mean(tf.pow(y_true_known - y_pred_known, 2)) czy tutaj nie powinno być axis=1 zamiast axis=2 ???
            loss = loss_miss

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        v = Visualizator(
            'result_' + str(self.params.method) + '_' + str(n_epochs) + '_' + str(self.params.num_sample) + '_' + str(
                self.gamma_int), 'loss', 100)
        train_loss = []
        with tf.Session() as sess:
            sess.run(init)  # run the initializer
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

                # print('Step {:d}: Minibatch Loss: {:.8f}, gamma: {:.4f}'.format(epoch, l, self.gamma.eval()))
            #v.plot_loss(train_loss, [i for i in range(n_epochs)], 'Koszt treningowy', 'koszt_treningowy')
            test_loss = []
            for i in range(10):
                if self.params.method != 'imputation':
                    batch_x = self.data_test[(i * self.params.nn):((i + 1) * self.params.nn), :]
                else:
                    batch_x = self.data_imputed_test[(i * self.params.nn):((i + 1) * self.params.nn), :]

                g, l_test = sess.run([decoder_op, loss], feed_dict={self.X: batch_x})
                # for j in range(self.params.nn):
                #     v.draw_mnist_image(i, j, g, self.params.method)
                test_loss.append(l_test)

            #v.plot_loss(test_loss, [i for i in range(10)], 'Koszt testowy', 'koszt_testowy')
        return np.mean(test_loss)


from sklearn.model_selection import ParameterGrid


def run_model():
    dataset_processor = DatasetProcessor(path='', dataset='mnist', width_mask=13, nn=100)
    X = dataset_processor.load_data()
    data_test, data_train = dataset_processor.divide_dataset_into_test_and_train(X)
    data_test = np.random.permutation(data_test)
    data_test, data_train = dataset_processor.change_background(data_test, data_train)
    data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data_imputed_train = imp.fit_transform(data_train)
    data_imputed_test = imp.transform(data_test)

    hyper_params = {'num_sample': [1, 5, 10], 'epoch': [150, 250], 'gamma': [0.0, 1.0, 2.0]}
    methods = ['first_layer', 'last_layer', 'different_cost']
    grid = ParameterGrid(hyper_params)
    f = open('loss_results_4', "a")
    for method in methods:
        for params in grid:
            print(method, params)
            p = AutoencoderFCParams(method=method, dataset='mnist', num_sample=params['num_sample'])
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


run_model()
