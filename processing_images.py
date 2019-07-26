import numpy as np
import os
import pathlib
import sys
from tensorflow.examples.tutorials.mnist import input_data


class FileProcessor:
    def __init__(self, path, type, nn, width_mask=13):
        self.path = path
        self.type = type
        self.x = None
        self.nn = nn
        self.data_test = None
        self.data_train = None
        self.labels = None
        self.width_mask = width_mask

    def read_data(self):
        if self.type == 'mnist':
            self.x = input_data.read_data_sets("./data_mnist/", one_hot=True)
        else:
            pass

    def random_mask_mnist(self, width_window, margin=0):
        margin_left = margin
        margin_righ = margin
        margin_top = margin
        margin_bottom = margin
        start_width = margin_top + np.random.randint(28 - width_window - margin_top - margin_bottom)
        start_height = margin_left + np.random.randint(28 - width_window - margin_left - margin_righ)

        return np.concatenate([28 * i + np.arange(start_height, start_height + width_window) for i in
                               np.arange(start_width, start_width + width_window)], axis=0).astype(np.int32)

    def data_with_mask_mnist(self, x, width_window=10):
        h = width_window
        for i in range(x.shape[0]):
            if width_window <= 0:
                h = np.random.randint(8, 20)
            mask = self.random_mask_mnist(h)
            x[i, mask] = np.nan
        return x

    def prepare_data(self):
        self.read_data()
        self.data_train = self.x.train.images
        self.labels = np.where(self.x.test.labels == 1)[1]
        data_test = self.x.test.images[np.where(self.labels == 0)[0][:self.nn], :]
        for i in range(1, 10):
            data_test = np.concatenate([self.data_test, self.x.test.images[np.where(self.labels == i)[0][:self.nn], :]],
                                       axis=0)

        self.data_test = np.random.permutation(data_test)
        self.change_background()
        self.mask_data()

    def change_background(self):
        self.data_train = 1. - self.data_train
        self.data_test = 1. - self.data_test

    def mask_data(self):
        self.data_train = self.data_with_mask_mnist(self.data_train, self.width_mask)
        self.data_test = self.data_with_mask_mnist(self.data_test, self.width_mask)

    def save_result_in_dir(self, save_dir, file_name):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(save_dir, file_name)).mkdir(parents=True, exist_ok=True)



