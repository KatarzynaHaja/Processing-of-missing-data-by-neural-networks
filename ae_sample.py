import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

class Sampling:
    def __init__(self, num_sample, params, x_miss, n_distribution):
        self.num_sample = num_sample  # 10
        self.params = params
        self.x_miss = x_miss
        self.size = tf.shape(x_miss)
        self.n_distribution = n_distribution

    def calculate_p(self, p,  means, covs, gamma):
        Q = []
        where_isfinite = tf.is_finite(self.x_miss)

        for i in range(self.n_distribution):
            norm = tf.subtract(self.x_miss, means[i, :])
            norm = tf.square(norm)
            q = tf.where(where_isfinite,
                         tf.reshape(tf.tile(tf.add(gamma, covs[i, :]), [self.size[0]]), [-1, self.size[1]]),
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

        return Q

    def random_component(self, p):
        return tf.squeeze(tf.gather(tf.random.multinomial(p, 1), 0))

    def random_from_component(self, mu, sigma):
        mvn = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma)
        return mvn.sample(1)

    def get_distibution_params(self, component, means, covs):
        where_isnan = tf.is_nan(self.x_miss)
        data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[component, :], [self.size[0]]), [-1, self.size[1]]), self.x_miss)
        miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[component, :], [self.size[0]]), [-1, self.size[1]]),
                            tf.zeros([self.size[0], self.size[1]]))

        return data_miss, miss_cov

    def generate_samples(self, p, x_miss, means, covs, num_input, gamma):
        size = tf.shape(x_miss)
        samples = tf.zeros([1, size[0], num_input])
        new_p = self.calculate_p(p, means, covs, gamma)

        for sam in range(self.num_sample):
            component = self.random_component(new_p)
            mu, sigma = self.get_distibution_params(component, means, covs)
            sample = self.random_from_component(mu, sigma)
            samples = tf.cond(tf.equal(tf.constant(sam), tf.constant(0)), lambda: tf.add(samples, sample),
                              lambda: tf.concat((samples, sample), axis=0))

        return samples

    def nr_after_first_layer(self, output):
        reshaped_output = tf.reshape(output, shape=(self.size[0] * self.num_sample, self.params.num_input))
        layer_1_m = tf.add(tf.matmul(reshaped_output, self.params.weights['encoder_h1']),
                           self.params.biases['encoder_b1'])
        layer_1_m = tf.nn.relu(layer_1_m)
        unreshaped = tf.reshape(layer_1_m, shape=(self.num_sample, self.size[0], self.params.num_hidden_1))
        mean = tf.reduce_mean(unreshaped, 0)
        return mean
