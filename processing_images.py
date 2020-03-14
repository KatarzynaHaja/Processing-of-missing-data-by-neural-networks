import numpy as np
import scipy.io as sio
from tensorflow.examples.tutorials.mnist import input_data
import os


class DatasetProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.x = None
        self.nn = 100 if self.dataset == 'mnist' else None
        self.width_mask = 13 if self.dataset == 'mnist' else 15
        self.size = 28 if dataset == 'mnist' else 32
        self.margin = 0

    def load_data(self):
        if self.dataset == 'mnist':
            return input_data.read_data_sets("./data_mnist/", one_hot=True)
        else:
            return self.get_svhn()

    def get_svhn(self):
        dataset_dir = "svhn_data/"
        filenames = ['train_32x32.mat', 'test_32x32.mat']
        if not all(os.path.isfile(dataset_dir + f) for f in filenames):
            raise ValueError(
                "No SVHN files in directory: {}. Files {} expected.".format(
                    dataset_dir, ", ".join(filenames)))
        dataset_train = sio.loadmat('svhn_data/train_32x32.mat')
        data_train = dataset_train["X"].transpose(3, 0, 1, 2)/255
        dataset_train['y'][np.where(dataset_train['y'] == 10 )] = 0
        labels_train = dataset_train['y']
        dataset_test = sio.loadmat('svhn_data/test_32x32.mat')
        data_test = dataset_test["X"].transpose(3, 0, 1, 2)/255
        dataset_test['y'][np.where(dataset_train['y'] == 10 )] = 0
        labels_test = dataset_train['y']

        return data_train, data_test, labels_train, labels_test

    def reshape_data_to_convolution(self, data):
        if self.dataset == 'mnist':
            return data.reshape(data.shape[0], 28, 28, 1)
        else:
            return data.reshape(data.shape[0], 32, 32, 3)

    def random_mask_mnist(self):
        margin_left = self.margin
        margin_righ = self.margin
        margin_top = self.margin
        margin_bottom = self.margin
        start_width = margin_top + np.random.randint(self.size - self.width_mask - margin_top - margin_bottom)
        start_height = margin_left + np.random.randint(self.size - self.width_mask - margin_left - margin_righ)

        return np.concatenate([self.size * i + np.arange(start_height, start_height + self.width_mask) for i in
                               np.arange(start_width, start_width + self.width_mask)], axis=0).astype(np.int32)

    def data_with_mask_mnist(self, x):
        for i in range(x.shape[0]):
            mask = self.random_mask_mnist()
            x[i, mask] = np.nan
        return x

    def divide_dataset_into_test_and_train(self, X):
        data_train = X.train.images
        labels = np.where(X.test.labels == 1)[1]
        data_test = X.test.images[np.where(labels == 0)[0][:self.nn], :]
        for i in range(1, 10):
            data_test = np.concatenate([data_test, X.test.images[np.where(labels == i)[0][:self.nn], :]],
                                       axis=0)

        return data_test, data_train

    def change_background(self, data_test, data_train):
        return 1. - data_test, 1. - data_train

    def mask_data(self, data_test, data_train):
        return self.data_with_mask_mnist(data_test), self.data_with_mask_mnist(data_train)
