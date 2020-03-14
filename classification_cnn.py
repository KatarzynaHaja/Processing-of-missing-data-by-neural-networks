import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import DatasetProcessor
from sampling_fc import Sampling
import numpy as np
import sys
import os

from visualization import Visualizator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ClassificationCNNParams:
    def __init__(self, method, dataset, num_sample=None, learning_rate=0.001):
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.width = 28
        self.length = 28
        self.num_channels = 1

        self.conv_layer_1_filters = [3, 3, 1, 32]
        self.max_pooling_1_ksize = 2
        self.max_pooling_1_stride = 2
        self.max_pooling_2_ksize = 2
        self.max_pooling_2_stride = 2
        self.conv_layer_2_filters = [3, 3, 32, 64]
        self.flatten_1 = [64 * 7 * 7, 128]
        self.flatten_2 = [128, 10]

        self.num_output = 10

        self.num_layers = 6

        self.nn = 100
        self.method = method
        self.dataset = dataset
        self.num_sample = num_sample
        self.learning_rate = learning_rate

        self.filters = [self.conv_layer_1_filters, self.conv_layer_2_filters]
        self.fully_connected = [self.flatten_1, self.flatten_2]

        self.filters_weights = []
        self.filters_biases = []
        self.fully_connected_weights = []
        self.fully_connected_biases = []

        # initilize weights and biases for filters
        for i in range(len(self.filters)):
            self.filters_weights.append(tf.Variable(initializer(self.filters[i])))
            self.filters_biases.append(tf.Variable(tf.random_normal([self.filters[i][-1]])))

        # initialize weights and biases for fully connected
        for i in range(len(self.fully_connected)):
            self.fully_connected_weights.append(tf.Variable(initializer(self.fully_connected[i])))
            self.fully_connected_biases.append(tf.Variable(tf.random_normal([self.fully_connected[i][-1]])))


class ClassificationCNN:
    def __init__(self, params, data_train, data_test, data_imputed_train, data_imputed_test, gamma, labels_train,
                 labels_test):
        self.data_train = data_train
        self.data_test = data_test
        self.data_imputed_train = data_imputed_train
        self.data_imputed_test = data_imputed_test
        self.params = params
        self.n_distribution = 5
        self.X = tf.placeholder("float", [None, self.params.width, self.params.length, self.params.num_channels])
        self.labels = tf.placeholder("float", [None, 10])
        self.gamma = gamma
        self.gamma_int = gamma

        self.gamma = tf.Variable(initial_value=self.gamma)

        self.labels_train = labels_train
        self.labels_test = labels_test

        # self.x_miss, self.x_known = self.divide_data_into_known_and_missing(self.X)
        # self.size = tf.shape(self.x_miss)

        self.size = tf.shape(self.X)

        # if self.params.method != 'imputation':
        #     self.sampling = Sampling(num_sample=self.params.num_sample, params=self.params, x_miss=self.x_miss,
        #                              n_distribution=self.n_distribution,
        #                              method=self.params.method)

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

    def model(self):
        conv_1 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=self.X, filters=self.params.filters_weights[0], strides=1, padding='SAME'),
                   self.params.filters_biases[0]))
        max_pooling_1 = tf.nn.max_pool2d(input=conv_1, ksize=self.params.max_pooling_1_ksize,
                                         strides=self.params.max_pooling_1_stride, padding='SAME')
        conv_2 = tf.nn.relu(
            tf.add(tf.nn.conv2d(input=max_pooling_1, filters=self.params.filters_weights[1], strides=1, padding='SAME'),
                   self.params.filters_biases[1]))

        max_pooling_2 = tf.nn.max_pool2d(input=conv_2, ksize=self.params.max_pooling_2_ksize,
                                         strides=self.params.max_pooling_2_stride, padding='SAME')

        reshaped_max_pooling_2 = tf.reshape(tensor=max_pooling_2, shape=(self.size[0], 7 * 7 * 64))

        layer_fc_1 = tf.nn.relu(
            tf.add(tf.matmul(reshaped_max_pooling_2, self.params.fully_connected_weights[0]),
                   self.params.fully_connected_biases[0]))

        # output:(?,128)

        layer_fc_2 = tf.add(tf.matmul(layer_fc_1, self.params.fully_connected_weights[1]),
                                       self.params.fully_connected_biases[1])

        # output : (?, 10)

        return layer_fc_2

    def main_loop(self, n_epochs):
        batch_size = 64

        loss = None

        self.set_variables()

        y_pred = self.model()  # prediction

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.labels,
            logits=y_pred
        )

        loss = tf.reduce_mean(loss)

        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.labels, 1),
                                          predictions=tf.argmax(y_pred, 1))

        optimizer = tf.train.RMSPropOptimizer(self.params.learning_rate).minimize(loss)

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
                        batch_x = self.data_imputed_train[(iteration * batch_size):((iteration + 1) * batch_size), :, :,
                                  :]

                    labels = self.labels_train[iteration * batch_size: (iteration + 1) * batch_size, :]
                    _, l, y = sess.run([optimizer, loss, y_pred], feed_dict={self.X: batch_x, self.labels: labels})
                print("Train loss", l)

                train_loss.append(l)

            test_loss = []
            for i in range(self.data_test.shape[0] // 2):
                if self.params.method != 'imputation':
                    batch_x = self.data_test[i * 2: (i + 1) * 2, :]
                else:
                    batch_x = self.data_imputed_test[i * 2: (i + 1) * 2, :]

                labels = self.labels_test[i * 2: (i + 1) * 2, :]

                accuracy, accuracy_op, tl = sess.run([acc, acc_op, loss],
                                                     feed_dict={self.X: batch_x, self.labels: labels})

            print("Accuracy:", accuracy_op, "Train loss:", tl)
            test_loss.append(tl)
        return accuracy_op, test_loss, train_loss


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

    data_imputed_train = dataset_processor.reshape_data_to_convolution(imp.fit_transform(data_train))
    data_imputed_test = dataset_processor.reshape_data_to_convolution(imp.transform(data_test))

    data_test = dataset_processor.reshape_data_to_convolution(data_test)
    data_train = dataset_processor.reshape_data_to_convolution(data_train)

    params = [{'method': 'imputation', 'params': [{'num_sample': 1, 'epoch': 250, 'gamma': 0.0}]}]
    f = open('loss_results_classification_conv', "a")
    for eleme in params:
        for param in eleme['params']:
            p = ClassificationCNNParams(method=eleme['method'], dataset='mnist', num_sample=param['num_sample'])
            a = ClassificationCNN(p, data_test=data_test, data_train=data_train, data_imputed_train=data_imputed_train,
                                  data_imputed_test=data_imputed_test,
                                  gamma=param['gamma'], labels_train=labels_train, labels_test=labels_test)
            accuracy = a.main_loop(param['epoch'])
            f.write(eleme['method'] + "," + str(param['num_sample']) + ','
                    + str(param['epoch']) + ',' + str(param['gamma']) + ',' + str(accuracy))
            f.write('\n')
    f.close()


run_model()
