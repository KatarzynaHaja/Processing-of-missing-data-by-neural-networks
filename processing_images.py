import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class DatasetProcessor:
    def __init__(self, path, dataset, nn, width_mask=13):
        self.path = path
        self.dataset = dataset
        self.x = None
        self.nn = nn
        self.width_mask = width_mask

    def load_data(self):
        if self.dataset == 'mnist':
            return input_data.read_data_sets("./data_mnist/", one_hot=True)
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
        return self.data_with_mask_mnist(data_test, self.width_mask), self.data_with_mask_mnist(data_train,
                                                                                                self.width_mask)
