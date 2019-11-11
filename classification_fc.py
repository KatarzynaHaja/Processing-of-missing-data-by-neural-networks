import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from ae_sample import Sampling
import numpy as np
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ClassificationFCParams:
    def __init__(self, method, dataset, num_sample=None, ):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.num_input = 784
        self.num_hidden_1 = 128  # 1st layer num features
        self.num_hidden_2 = 64  # 2nd layer num features (the latent dim)
        self.num_output = 10

        self.nn = 100
        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample

        self.weights = {
            'h1': tf.Variable(initializer([self.num_input, self.num_hidden_1])),
            'h2': tf.Variable(initializer([self.num_hidden_1, self.num_hidden_2])),
            'h3': tf.Variable(initializer([self.num_hidden_2, self.num_output])),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.num_output])),
        }


class ClassificationFC:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma, labels_train,
                 labels_test):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
        self.n_distribution = 5
        self.X = tf.placeholder("float", [None, self.params.num_input])
        self.labels = tf.placeholder("float", [None, 10])
        self.gamma = gamma
        self.gamma_int = gamma

        if self.params.method != 'theirs':
            self.gamma = tf.Variable(initial_value=self.gamma)
        else:
            self.gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)

        self.labels_train = labels_train
        self.labels_test = labels_test

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

        x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))
        x_known = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))

        return x_miss, x_known

    def model(self):
        if self.params.method != 'imputation':
            if self.params.method == 'theirs':
                gamma = tf.abs(self.gamma)
                gamma_ = tf.cond(tf.less(gamma[0], 1.), lambda: gamma, lambda: tf.pow(gamma, 2))
                covs = tf.abs(self.covs)
                p = tf.abs(self.p)
                p = tf.div(p, tf.reduce_sum(p, axis=0))

                layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x_known, self.params.weights['h1']),
                                            self.params.biases['b1']))

                where_isnan = tf.is_nan(self.x_miss)
                where_isfinite = tf.is_finite(self.x_miss)

                weights2 = tf.square(self.params.weights['h1'])

                Q = []
                layer_1_miss = tf.zeros([self.size[0], self.params.num_hidden_1])
                for i in range(self.n_distribution):
                    data_miss = tf.where(where_isnan,
                                         tf.reshape(tf.tile(self.means[i, :], [self.size[0]]), [-1, self.size[1]]),
                                         self.x_miss)
                    miss_cov = tf.where(where_isnan,
                                        tf.reshape(tf.tile(covs[i, :], [self.size[0]]), [-1, self.size[1]]),
                                        tf.zeros([self.size[0], self.size[1]]))

                    layer_1_m = tf.add(tf.matmul(data_miss, self.params.weights['h1']),
                                       self.params.biases['b1'])

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
                                 tf.reshape(tf.tile(tf.add(gamma_, covs[i, :]), [self.size[0]]), [-1, self.size[1]]),
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
                layer_1_miss = tf.reshape(layer_1_miss,
                                          shape=(self.n_distribution, self.size[0], self.params.num_hidden_1))
                layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)

                layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)
            else:

                samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs,
                                                         self.params.num_input,
                                                         self.gamma)
                layer_1 = tf.nn.relu(
                    tf.add(tf.matmul(self.x_known, self.params.weights['h1']),
                           self.params.biases['b1']))

                layer_1_miss = self.sampling.nr(samples)
                layer_1_miss = tf.reshape(layer_1_miss, shape=(self.size[0], self.params.num_hidden_1))

                layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)

        if self.params.method == 'imputation':
            layer_1 = tf.nn.relu(
                tf.add(tf.matmul(self.X, self.params.weights['h1']),
                       self.params.biases['b1']))

        layer_2 = tf.nn.relu(
            tf.add(tf.matmul(layer_1, self.params.weights['h2']), self.params.biases['b2']))

        layer_3 = tf.add(tf.matmul(layer_2, self.params.weights['h3']), self.params.biases['b3'])

        if self.params.method != 'last_layer':
            return layer_3

        if self.params.method == 'last_layer':
            input = layer_3[:self.size[0] * self.params.num_sample, :]
            mean = self.sampling.mean_sample(input, self.params.num_output)
            return mean

        return layer_3

    def main_loop(self, n_epochs):
        learning_rate = 0.01
        batch_size = 64

        loss = None

        self.set_variables()

        y_pred = self.model()  # prediction

        if self.params.method != 'different_cost':
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels,
                logits=y_pred
            )

            loss = tf.reduce_mean(loss)

            acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.labels, 1),
                                              predictions=tf.argmax(y_pred, 1))

        if self.params.method == 'different_cost':
            labels = tf.expand_dims(self.labels, 0)
            labels = tf.tile(labels, [self.params.num_sample, 1, 1])
            labels = tf.reshape(labels, shape=(self.params.num_sample * self.size[0], 10))

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=y_pred,
            ))

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        init_loc = tf.local_variables_initializer()

        train_loss = []
        with tf.Session() as sess:
            sess.run(init)
            sess.run(init_loc)
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

                    labels = self.labels_train[iteration * batch_size: (iteration + 1) * batch_size, :]
                    _, l = sess.run([optimizer, loss], feed_dict={self.X: batch_x, self.labels: labels})

                train_loss.append(l)

            for i in range(self.data_test.shape[0] // 2):
                if self.params.method != 'imputation':
                    batch_x = self.data_test[i * 2: (i + 1) * 2, :]
                else:
                    batch_x = self.data_imputed_test[i * 2: (i + 1) * 2, :]

                labels = self.labels_test[i * 2: (i + 1) * 2, :]

                accuracy, accuracy_op, y = sess.run([acc, acc_op, y_pred],
                                                    feed_dict={self.X: batch_x, self.labels: labels})

            print(accuracy_op)
        return accuracy_op


from sklearn.model_selection import ParameterGrid


def run_model():
    dataset_processor = DatasetProcessor(dataset='mnist')
    X = dataset_processor.load_data()
    data_train = X.train.images
    data_test = X.test.images
    labels_train = X.train.labels
    labels_test = X.test.labels

    data_test, data_train = dataset_processor.change_background(data_test, data_train)
    data_test, data_train = dataset_processor.mask_data(data_test, data_train)

    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data_imputed_train = imp.fit_transform(data_train).reshape(data_train.shape[0], 784)
    data_imputed_test = imp.transform(data_test).reshape(data_test.shape[0], 784)

    hyper_params = {'num_sample': [20], 'epoch': [250], 'gamma': [0.0]}
    methods = ['last_layer', 'theirs', 'imputation']
    grid = ParameterGrid(hyper_params)
    f = open('loss_results_classification', "a")
    for method in methods:
        for params in grid:
            p = ClassificationFCParams(method=method, dataset='mnist', num_sample=params['num_sample'])
            a = ClassificationFC(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                                 data_imputed_test=data_imputed_test,
                                 gamma=params['gamma'], labels_train=labels_train, labels_test=labels_test)
            accuracy = a.main_loop(params['epoch'])
            f.write(method + "," + str(params['num_sample']) + ','
                    + str(params['epoch']) + ',' + str(params['gamma']) + ',' + str(accuracy))
            f.write('\n')
    f.close()


run_model()
