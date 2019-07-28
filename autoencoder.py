import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
from processing_images import FileProcessor

class AutoencoderParams:
    def __init__(self):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.num_hidden_1 = 256  # 1st layer num features
        self.num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_3 = 64  # 3nd layer num features (the latent dim)
        self. num_input = 784  # MNIST data_rbfn input (img shape: 28*28)

        self.X = tf.placeholder("float", [None, self.num_input])

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



class Autoencoder:
    def __init__(self, params, method):
        self.params = params
        self.method = method
        self.file_processor = FileProcessor(path='', type= 'mnist', width_mask=13, nn=10)
        self.data_train = None
        self.data_test = None
        self.n_distribution = 5  # number of n_distribution


    def load_data(self):
        self.data_train, self.data_test = self.file_processor.prepare_data()

    def set_variables(self):
        imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
        data = imp.fit_transform(self.data_train)
        gmm = GaussianMixture(n_components=self.n_distribution, covariance_type='diag').fit(data)
        self.p = tf.Variable(initial_value=gmm.weights_.reshape((-1, 1)), dtype=tf.float32)
        self.means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
        self.covs = tf.abs(tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32))
        self.gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)



    def calculate_p(self):
        pass

    def prepare_data(self, x):
        check_isnan = tf.is_nan(x)
        check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

        x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0), [-1]))  # data with missing values
        x_known = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))  # data_without missing values

        return x_miss, x_known

    def encoder(self,x):

        x_miss, x_known = self.prepare_data(x)
        # data without missing
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x_known,self.params. weights['encoder_h1']), self.params.biases['encoder_b1']))
        size = tf.shape(x_miss)
        samples = create_data(num_sample, p_, x_miss, means, covs)
        layer_1_miss = nr(samples, size, num_sample)
        layer_1_miss = tf.reshape(layer_1_miss, shape=(size[0], self.params.num_hidden_1))

        layer_1 = tf.concat((layer_1_miss, layer_1), axis=0)

        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.params.weights['encoder_h2']), self.params.biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.params.weights['encoder_h3']), self.params.biases['encoder_b3']))
        return layer_3, size

    # Building the decoder
    def decoder(self, x, size):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.params.weights['decoder_h1']), self.params.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.params.weights['decoder_h2']), self.params.biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,self.params.weights['decoder_h3']), self.params.biases['decoder_b3']))
        return layer_3

def autoencoder_main_loop():
    learning_rate = 0.01
    n_epochs = 250
    batch_size = 64
