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

    def get_svhn(extra=False):
        dataset_dir = "svhn_data/"
        filenames = ['train_32x32.mat', 'test_32x32.mat']
        if not all(os.path.isfile(dataset_dir + f) for f in filenames):
            raise ValueError(
                "No SVHN files in directory: {}. Files {} expected.".format(
                    dataset_dir, ", ".join(filenames)))
        dataset_train = sio.loadmat('svhn_data/train_32x32.mat')
        dataset_train = {'X': dataset_train["X"].transpose(3, 0, 1, 2).astype(np.float32),
                         'Y': [0 if i == 10 else i for i in dataset_train["y"]]}

        if extra:
            dataset_extra = sio.loadmat('svhn_data/extra_32x32.mat')
            dataset_extra = {'X': dataset_extra["X"].transpose(3, 0, 1, 2).astype(np.float32),
                             'Y': [0 if i == 10 else i for i in dataset_extra["y"]]}
            dataset_train = {'X': np.concatenate([dataset_train['X'], dataset_extra['X']], axis=0),
                             'Y': np.concatenate([dataset_train['Y'], dataset_extra['Y']], axis=0)}

        dataset_test = sio.loadmat('svhn_data/test_32x32.mat')
        dataset_test = {'X': dataset_test["X"].transpose(3, 0, 1, 2).astype(np.float32),
                        'Y': [0 if i == 10 else i for i in dataset_test["y"]]}

        return dataset_train, dataset_test

    def reshape_data(self, data):
        return data.reshape(data.shape[0], 32*32 *3)

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
        if self.dataset == 'svhn':
            x_with_labels = x
            x = x['X']
        for i in range(x.shape[0]):
            mask = self.random_mask_mnist()
            x[i, mask] = np.nan
        if self.dataset == 'svhn':
            x = {'X': x, 'Y': x_with_labels['Y']}
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
