import argparse
import os
import sys
from time import time

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

import tensorflow_probability as tfp

tfd = tfp.distributions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


class Helper:

    @classmethod
    def read_data(self, path, sep=',', val_type='f8'):
        return np.genfromtxt(path, dtype=val_type, delimiter=sep)

    @classmethod
    def scaler_range(self, X, feature_range=(-1, 1), min_x=None, max_x=None):
        if min_x is None:
            min_x = np.nanmin(X, axis=0)
            max_x = np.nanmax(X, axis=0)

        X_std = (X - min_x) / (max_x - min_x)
        X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_scaled, min_x, max_x

    @classmethod
    def prep_labels(self, x, y):
        check_isnan = tf.is_nan(x)
        check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

        y_miss = tf.gather(y, tf.reshape(tf.where(check_isnan > 0), [-1]))
        y = tf.gather(y, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
        return tf.concat((y, y_miss), axis=0)

    @classmethod
    def nr_one_sample(self, mu, sigma):
        mvn = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma)
        return mvn.sample(1)

    @classmethod
    def nr_multi_sample(self, mu, sigma):
        mvn = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma)
        samples = mvn.sample(10)
        return samples


class Model:
    def __init__(self, x, means, covs, p, gamma, weights, biases):
        self.weights = weights
        self.biases = biases
        self.means = means
        self.covs = covs
        self.x = x
        self.gamma_ = tf.abs(gamma)
        self.p = p
        self.covs_ = tf.abs(self.covs)
        self.p_ = tf.nn.softmax(self.p)

    def initialize(self):
        self.chosen_component = tf.squeeze(tf.gather(tf.random.multinomial(self.p_, 1), 0))

        check_isnan = tf.reduce_sum(tf.cast(tf.is_nan(self.x), tf.int32), 1)

        self.x_miss = tf.gather(self.x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
        self.x = tf.gather(self.x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data without missing values
        # data without missing
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1']))

        # data with missing
        self.where_isnan = tf.is_nan(self.x_miss)
        self.where_isfinite = tf.is_finite(self.x_miss)
        self.size = tf.shape(self.x_miss)

        self.weights2 = tf.square(self.weights['h1'])

    def calculate_component(self, chosen_component):
        data_miss = tf.where(self.where_isnan, tf.reshape(tf.tile(self.means[chosen_component, :], [self.size[0]]), [-1, self.size[1]]),
                             self.x_miss)
        miss_cov = tf.where(self.where_isnan, tf.reshape(tf.tile(self.covs_[chosen_component, :], [self.size[0]]), [-1, self.size[1]]),
                            tf.zeros([self.size[0], self.size[1]]))


        output = Helper.nr_one_sample(data_miss, miss_cov)
        print(output)

        layer_1_m = tf.nn.relu(tf.add(tf.matmul(output, self.weights['h1']), self.biases['b1']))

        norm = tf.subtract(data_miss, self.means[chosen_component, :])
        norm = tf.square(norm)
        q = tf.where(self.where_isfinite,
                     tf.reshape(tf.tile(tf.add(self.gamma_, self.covs_[chosen_component, :]), [self.size[0]]), [-1, self.size[1]]),
                     tf.ones_like(self.x_miss))
        norm = tf.div(norm, q)
        norm = tf.reduce_sum(norm, axis=1)

        q = tf.log(q)
        q = tf.reduce_sum(q, axis=1)

        q = tf.add(q, norm)

        norm = tf.cast(tf.reduce_sum(tf.cast(self.where_isfinite, tf.int32), axis=1), tf.float32)
        norm = tf.multiply(norm, tf.log(2 * np.pi))

        q = tf.add(q, norm)
        q = -0.5 * q

        return layer_1_m, q

    def model(self):
        final_distributions, final_q = self.calculate_component(self.chosen_component)

        distrib = final_distributions
        log_q = final_q

        log_q = tf.add(log_q, tf.log(self.p_))
        r = tf.nn.softmax(log_q, axis=0)

        layer_1_miss = tf.multiply(distrib, r[:, :, tf.newaxis])
        layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)

        # join layer for data_rbfn with missing values with layer for data_rbfn without missing values
        layer_1 = tf.concat((self.layer_1, layer_1_miss), axis=0)

        # Encoder Hidden layer with sigmoid activation
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4']))
        return tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./', help='Directory for input data')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--training_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--n_distribution', type=int, default=3, help='Number of distributions')
    FLAGS, unparsed = parser.parse_known_args()
    # print([sys.argv[0]] + unparsed)
    path_dir = FLAGS.data_dir

    # Parameters
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    training_epochs = FLAGS.training_epochs

    # Network Parameters
    n_distribution = FLAGS.n_distribution

    data = Helper.read_data(os.path.join(path_dir, '_data.txt'))
    data, minx, maxx = Helper.scaler_range(data, feature_range=(-1, 1))

    labels = Helper.read_data(os.path.join(path_dir, '_labels.txt'))
    lb = LabelBinarizer()
    lb.fit(labels)

    class_name = lb.classes_
    n_class = class_name.shape[0]
    if n_class == 2:
        lb.fit(np.append(labels, np.max(class_name) + 1))

    n_features = data.shape[1]
    num_hidden_1 = int(0.5 * n_features)
    num_hidden_2 = num_hidden_1
    num_hidden_3 = num_hidden_1
    num_hidden_4 = num_hidden_1
    num_hidden_5 = n_class

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    complate_data = imp.fit_transform(data)
    gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(complate_data)
    del complate_data, imp

    gmm_weights = np.log(gmm.weights_.reshape((-1, 1)))
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_
    del gmm

    acc = np.zeros((3, 5))

    time_train = np.zeros(5)
    time_test = np.zeros(5)

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    for trn_index, test_index in skf.split(data, labels):
        X_train = data[trn_index]
        X_lab = labels[trn_index]
        train_index, valid_index = next(StratifiedKFold(n_splits=5).split(X_train, X_lab))

        train_x = X_train[train_index, :]
        valid_x = X_train[valid_index, :]
        test_x = data[test_index, :]

        train_y = lb.transform(X_lab[train_index])
        valid_y = lb.transform(X_lab[valid_index])
        test_y = lb.transform(labels[test_index])
        if n_class == 2:
            train_y = train_y[:, :-1]
            valid_y = valid_y[:, :-1]
            test_y = test_y[:, :-1]

        with tf.Graph().as_default() as graph:

            initializer = tf.contrib.layers.variance_scaling_initializer()

            weights = {
                'h1': tf.Variable(initializer([n_features, num_hidden_1])),
                'h2': tf.Variable(initializer([num_hidden_1, num_hidden_2])),
                'h3': tf.Variable(initializer([num_hidden_2, num_hidden_3])),
                'h4': tf.Variable(initializer([num_hidden_3, num_hidden_4])),
                'h5': tf.Variable(initializer([num_hidden_4, num_hidden_5])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([num_hidden_1])),
                'b2': tf.Variable(tf.random_normal([num_hidden_2])),
                'b3': tf.Variable(tf.random_normal([num_hidden_3])),
                'b4': tf.Variable(tf.random_normal([num_hidden_4])),
                'b5': tf.Variable(tf.random_normal([num_hidden_5])),
            }

            # Symbols
            z = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
            y = tf.placeholder(shape=[None, n_class], dtype=tf.float32)

            p = tf.Variable(initial_value=gmm_weights, dtype=tf.float32)

            means = tf.Variable(initial_value=gmm_means, dtype=tf.float32)
            covs = tf.Variable(initial_value=gmm_covariances, dtype=tf.float32)

            gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=2, stddev=1.), dtype=tf.float32)

            # Construct model
            m = Model(z, means, covs, p, gamma, weights, biases)
            m.initialize()
            predict = m.model()

            y_true = Helper.prep_labels(z, y)

            # Mean squared error
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y_true))

            l_r = learning_rate
            # Gradient descent
            optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()

            nr_epoch = 10

            val_weights = None
            val_biases = None
            val_p = None
            val_means = None
            val_covs = None
            val_gamma = None

            with tf.Session(graph=graph).as_default() as sess:
                sess.run(init)
                min_cost = np.inf
                n_cost_up = 0

                prev_train_cost = np.inf

                time_train[id_acc] = time()

                epoch = 0
                # Training cycle
                for epoch in range(training_epochs):
                    # print("\r[{}|{}] Step: {:d} from 5".format(epoch + 1, training_epochs, id_acc), end="")
                    # sys.stdout.flush()

                    curr_train_cost = []
                    for batch_idx in range(0, train_y.shape[0], batch_size):
                        x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                        y_batch = train_y[batch_idx:batch_idx + batch_size, :]

                        temp_train_cost, _ = sess.run([cost, optimizer], feed_dict={z: x_batch, y: y_batch})
                        curr_train_cost.append(temp_train_cost)

                    curr_train_cost = np.asarray(curr_train_cost).mean()

                    if epoch > nr_epoch and (prev_train_cost - curr_train_cost) < 1e-4 < l_r:
                        l_r = l_r / 2.
                        optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

                    prev_train_cost = curr_train_cost

                    curr_cost = []
                    for batch_idx in range(0, valid_y.shape[0], batch_size):
                        x_batch = valid_x[batch_idx:batch_idx + batch_size, :]
                        y_batch = valid_y[batch_idx:batch_idx + batch_size, :]
                        curr_cost.append(sess.run(cost, feed_dict={z: x_batch, y: y_batch}))

                    curr_cost = np.asarray(curr_cost).mean()

                    if min_cost > curr_cost:
                        min_cost = curr_cost
                        n_cost_up = 0

                        val_weights = {
                            'h1': weights['h1'].eval(),
                            'h2': weights['h2'].eval(),
                            'h3': weights['h3'].eval(),
                            'h4': weights['h4'].eval(),
                            'h5': weights['h5'].eval(),
                        }
                        val_biases = {
                            'b1': biases['b1'].eval(),
                            'b2': biases['b2'].eval(),
                            'b3': biases['b3'].eval(),
                            'b4': biases['b4'].eval(),
                            'b5': biases['b5'].eval(),
                        }

                        val_p = p.eval()
                        val_means = means.eval()
                        val_covs = covs.eval()
                        val_gamma = gamma.eval()
                    elif epoch > nr_epoch:
                        n_cost_up = n_cost_up + 1

                    if n_cost_up == 5 and 1e-4 < l_r:
                        l_r = l_r / 2.
                        optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)
                    elif n_cost_up == 10:
                        break

                time_train[id_acc] = (time() - time_train[id_acc]) / (epoch + 1)

                means.load(val_means)
                covs.load(val_covs)
                p.load(val_p)
                gamma.load(val_gamma)

                for key in weights.keys():
                    weights[key].load(val_weights[key])
                for key in biases.keys():
                    biases[key].load(val_biases[key])

                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                train_accuracy = []
                for batch_idx in range(0, train_y.shape[0], batch_size):
                    x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = train_y[batch_idx:batch_idx + batch_size, :]

                    train_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                train_accuracy = np.mean(train_accuracy)

                valid_accuracy = []
                for batch_idx in range(0, valid_y.shape[0], batch_size):
                    x_batch = valid_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = valid_y[batch_idx:batch_idx + batch_size, :]

                    valid_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                valid_accuracy = np.mean(valid_accuracy)

                time_test[id_acc] = time()
                test_accuracy = []
                for batch_idx in range(0, test_y.shape[0], batch_size):
                    x_batch = test_x[batch_idx:batch_idx + batch_size, :]
                    y_batch = test_y[batch_idx:batch_idx + batch_size, :]
                    test_accuracy.append(accuracy.eval({z: x_batch, y: y_batch}))
                test_accuracy = np.mean(test_accuracy)
                time_test[id_acc] = time() - time_test[id_acc]

                acc[0, id_acc] = train_accuracy
                acc[1, id_acc] = valid_accuracy
                acc[2, id_acc] = test_accuracy
                id_acc = id_acc + 1

    mean_acc = np.average(acc, axis=1)
    std_acc = np.std(acc, axis=1)
    sys.stdout.flush()

    print(
        "Mean acc{:.4f}; Std acc{:.4f}; mean acc 1 {:.4f}; std acc 1 {:.4f}; mean acc 2 {:.4f}; std acc 2 {:.4f}; time train{:.4f}; time test{:.4f};{};{};{};{}".format(
            mean_acc[0], std_acc[0], mean_acc[1], std_acc[1], mean_acc[2], std_acc[2], np.average(time_train),
            np.average(time_test), FLAGS.learning_rate, FLAGS.batch_size, FLAGS.training_epochs, FLAGS.n_distribution))


if __name__ == '__main__':
    main()
