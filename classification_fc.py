import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from ae_sample import Sampling
import numpy as np
import sys
import os
import analitic_method

from visualization import Visualizator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ClassificationFCParams:
    def __init__(self, method, dataset, num_sample=None):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.num_input = 784
        self.num_hidden_1 = 512
        self.num_hidden_2 = 256  # 1st layer num features
        self.num_hidden_3 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_4 = 64
        self.num_output = 10

        self.n_distribution = 5

        self.num_layers = 4

        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample

        self.layer_inputs = [self.num_input, self.num_hidden_1, self.num_hidden_2, self.num_hidden_3, self.num_hidden_4]
        self.layer_outputs = [self.num_hidden_1, self.num_hidden_2, self.num_hidden_3, self.num_hidden_4,
                              self.num_output]


        self.weights = []
        self.biases = []
        for i in range(self.num_layers + 1):
            tf.Variable(initializer([self.num_input, self.num_hidden_1]))
            self.weights.append(tf.Variable(initializer([self.layer_inputs[i], self.layer_outputs[i]])))
            self.biases.append(tf.Variable(tf.random_normal([self.layer_outputs[i]])))


class ClassificationFC:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma, labels_train,
                 labels_test):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
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
                                     n_distribution=self.params.n_distribution,
                                     method=self.params.method)

    def set_variables(self):
        if self.params.method != 'imputation':
            gmm = GaussianMixture(n_components=self.params.n_distribution, covariance_type='diag').fit(self.data_imputed_train)
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
                layer_1 = analitic_method.nr(self.gamma, self.covs, self.p, self.params, self.means,
                                             self.x_miss, self.x_known, self.params.weights[0],
                                             self.params.biases[0])
            else:
                layer_1 = tf.nn.relu(
                    tf.add(tf.matmul(self.x_known, self.params.weights[0]),
                           self.params.biases[0]))
                samples = self.sampling.generate_samples(self.p, self.x_miss, self.means, self.covs,
                                                         self.params.num_input,
                                                         self.gamma)

                layer_1_miss = self.sampling.nr(samples, self.params.weights[0],
                                                self.params.biases[0])
                layer_1_miss = tf.reshape(layer_1_miss, shape=(self.size[0], self.params.num_hidden_1))

                layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)

        if self.params.method == 'imputation':
            layer_1 = tf.nn.relu(
                    tf.add(tf.matmul(self.X, self.params.weights[0]),
                           self.params.biases[0]))

        layer = layer_1

        for i in range(self.params.num_layers-1):
            layer = tf.nn.relu(
                tf.add(tf.matmul(layer, self.params.weights[i+1]), self.params.biases[i+1]))

        layer = tf.add(tf.matmul(layer, self.params.weights[-1]), self.params.biases[-1])


        if self.params.method != 'last_layer':
            return layer

        if self.params.method == 'last_layer':
            input = layer[:self.size[0] * self.params.num_sample, :]
            mean = self.sampling.mean_sample(input, self.params.num_output)
            return mean

        return layer

    def main_loop(self, n_epochs):
        learning_rate = 0.001
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

            acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.labels, 1),
                                              predictions=tf.argmax(y_pred, 1))

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        init_loc = tf.local_variables_initializer()

        v = Visualizator(
            'result_' + str(self.params.method) + '_' + str(n_epochs) + '_' + str(self.params.num_sample) + '_' + str(
                self.gamma_int), 'loss')

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
                    _, l, y = sess.run([optimizer, loss, y_pred], feed_dict={self.X: batch_x, self.labels: labels})

                print('Loss:', l)

                train_loss.append(l)

            for i in range(self.data_test.shape[0] // 2):
                if self.params.method != 'imputation':
                    batch_x = self.data_test[i * 2: (i + 1) * 2, :]
                else:
                    batch_x = self.data_imputed_test[i * 2: (i + 1) * 2, :]

                labels = self.labels_test[i * 2: (i + 1) * 2, :]

                accuracy, accuracy_op, test_loss = sess.run([acc, acc_op, loss],
                                                            feed_dict={self.X: batch_x, self.labels: labels})

            print("Accuracy:", accuracy_op, "Train loss:", test_loss)
        return accuracy_op


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

    params = [
        {'method': 'theirs', 'params': [{'num_sample': 1, 'epoch': 250, 'gamma': 0.0}]},
        # {'method': 'first_layer', 'params': [{'num_sample': 10, 'epoch': 250, 'gamma': 1.5}]},
        {'method': 'last_layer', 'params': [{'num_sample': 10, 'epoch': 250, 'gamma': 0.0}]},
        #                                     {'num_sample': 20, 'epoch': 250, 'gamma': 0.0},
        #                                     {'num_sample': 100, 'epoch': 150, 'gamma': 1.0}]},
        {'method': 'imputation', 'params': [{'num_sample': 1, 'epoch': 100, 'gamma': 0.0}]}
        # {'method': 'different_cost', 'params': [{'num_sample': 10, 'epoch': 250, 'gamma': 0.5}]}

    ]
    f = open('loss_results_classification_fc', "a")
    for eleme in params:
        for param in eleme['params']:
            p = ClassificationFCParams(method=eleme['method'], dataset='mnist', num_sample=param['num_sample'])
            a = ClassificationFC(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                                 data_imputed_test=data_imputed_test,
                                 gamma=param['gamma'], labels_train=labels_train, labels_test=labels_test)
            accuracy = a.main_loop(param['epoch'])
            f.write(eleme['method'] + "," + str(param['num_sample']) + ','
                    + str(param['epoch']) + ',' + str(param['gamma']) + ',' + str(accuracy))
            f.write('\n')
    f.close()


run_model()
