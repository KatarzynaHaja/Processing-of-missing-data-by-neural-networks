# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# X = input_data.read_data_sets("./data_mnist/", one_hot=True)
# image = X.train.images[0]
# plt.imshow(image.reshape(28, 28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example',  bbox_inches='tight', pad_inches=0)
#
# new_image = 1- image
# plt.imshow(new_image.reshape(28,28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example_changed_background',  bbox_inches='tight', pad_inches=0)
#
# import numpy as np
#
#
# def random_mask_mnist(width_window, margin=0):
#     margin_left = margin
#     margin_righ = margin
#     margin_top = margin
#     margin_bottom = margin
#     start_width = margin_top + np.random.randint(28 - width_window - margin_top - margin_bottom)
#     start_height = margin_left + np.random.randint(28 - width_window - margin_left - margin_righ)
#
#     return np.concatenate([28 * i + np.arange(start_height, start_height + width_window) for i in
#                            np.arange(start_width, start_width + width_window)], axis=0).astype(np.int32)
#
#
# def data_with_mask_mnist(x, width_window=10):
#     h = width_window
#     if width_window <= 0:
#         h = np.random.randint(8, 20)
#     mask = random_mask_mnist(h)
#     x[mask] = 1
#     return x
#
# image_with_patch = data_with_mask_mnist(new_image, 13)
#
# plt.imshow(new_image.reshape(28,28), origin="upper", cmap="gray")
# plt.axis('off')
# plt.savefig('mnist_example_with_patch',  bbox_inches='tight', pad_inches=0)

#
# from processing_images import DatasetProcessor
# import matplotlib.pyplot as plt
#
# d = DatasetProcessor('svhn')
# data_train, data_test = d.load_data()
# data_train['X'] = d.reshape_data(data_train['X'])
# data_test['X'] = d.reshape_data(data_test['X'])
# data_train, data_test = d.mask_data(data_train, data_test)
#
# plt.imshow(data_train['X'][0].reshape(32, 32,3).astype('uint8'), origin="upper")
# plt.axis('off')
# plt.savefig('svhn_masked_example',  bbox_inches='tight', pad_inches=0)

import numpy as np
data = [0.0052325632, 0.008187881, 0.014886782]
labels = ['metoda analiczna', 'ostatnia warstwa', 'imputacja']
pos = np.arange(len(data))

plt.bar(pos,data,color='lightsteelblue', width=0.3)
plt.xticks(pos, labels)
plt.xlabel('Metody', fontsize=12)
plt.ylabel('Średni koszt', fontsize=12)
plt.title('Porównianie średniego kosztu - autoenkoder liniowy', fontsize=13)
plt.savefig("Porównianie średniego kosztu")
plt.show()



