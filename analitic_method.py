import tensorflow as tf
import numpy as np


def nr(gamma, covs, p, params, means, x_miss, x_known,weights, bias):
    gamma = tf.abs(gamma)
    gamma_ = tf.cond(tf.less(gamma[0], 1.), lambda: gamma, lambda: tf.pow(gamma, 2))
    covs = tf.abs(covs)
    p = tf.abs(p)
    p = tf.div(p, tf.reduce_sum(p, axis=0))
    size = tf.shape(x_miss)

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x_known,weights),bias))

    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)

    weights2 = tf.square(weights)

    Q = []
    layer_1_miss = tf.zeros([size[0], params.num_hidden_1])
    for i in range(params.n_distribution):
        data_miss = tf.where(where_isnan, tf.reshape(tf.tile(means[i, :], [size[0]]), [-1, size[1]]),
                             x_miss)
        miss_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs[i, :], [size[0]]), [-1, size[1]]),
                            tf.zeros([size[0], size[1]]))

        layer_1_m = tf.add(tf.matmul(data_miss, weights),bias)

        layer_1_m = tf.div(layer_1_m, tf.sqrt(tf.matmul(miss_cov, weights2)))
        layer_1_m = tf.div(tf.exp(tf.div(-tf.pow(layer_1_m, 2), 2.)), np.sqrt(2 * np.pi)) + tf.multiply(
            tf.div(layer_1_m, 2.), 1 + tf.erf(
                tf.div(layer_1_m, np.sqrt(2))))

        layer_1_miss = tf.cond(tf.equal(tf.constant(i), tf.constant(0)),
                               lambda: tf.add(layer_1_miss, layer_1_m),
                               lambda: tf.concat((layer_1_miss, layer_1_m), axis=0))

        norm = tf.subtract(data_miss, means[i, :])
        norm = tf.square(norm)
        q = tf.where(where_isfinite,
                     tf.reshape(tf.tile(tf.add(gamma_, covs[i, :]), [size[0]]), [-1, size[1]]),
                     tf.ones_like(x_miss))
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

    Q = tf.reshape(Q, shape=(params.n_distribution, -1))
    Q = tf.add(Q, tf.log(p))
    Q = tf.subtract(Q, tf.reduce_max(Q, axis=0))
    Q = tf.where(Q < -20, tf.multiply(tf.ones_like(Q), -20), Q)
    Q = tf.exp(Q)
    Q = tf.div(Q, tf.reduce_sum(Q, axis=0))
    Q = tf.reshape(Q, shape=(-1, 1))

    layer_1_miss = tf.multiply(layer_1_miss, Q)
    layer_1_miss = tf.reshape(layer_1_miss, shape=(params.n_distribution, size[0], params.num_hidden_1))
    layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)
    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)

    return layer_1