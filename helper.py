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

# data = [0.0052325632, 0.008187881, 0.014886782]
# labels = ['metoda analiczna', 'ostatnia warstwa', 'imputacja']
# pos = np.arange(len(data))
#
# plt.bar(pos,data,color='lightsteelblue', width=0.3)
# plt.xticks(pos, labels)
# plt.xlabel('Metody', fontsize=12)
# plt.ylabel('Średni koszt', fontsize=12)
# plt.title('Porównianie średniego kosztu - autoenkoder liniowy', fontsize=13)
# plt.savefig("Porównianie średniego kosztu")
# plt.show()


# data = [0.8334, 0.8414,0.7650]
# labels = ['metoda analiczna', 'ostatnia warstwa', 'imputacja']
# pos = np.arange(len(data))
#
# plt.bar(pos,data,color='lightsteelblue', width=0.3)
# plt.xticks(pos, labels)
# plt.xlabel('Metody', fontsize=12)
# plt.ylabel('Precyzja', fontsize=12)
# plt.title('Porównianie precyzji - klasyfikator z warstwami liniowymi', fontsize=13)
# plt.savefig("Porównianie precyzji klasyfikator liniowy")
# plt.show()

results = {
    'metoda analityczna': [0.88324475, 0.6887004, 0.60074747, 0.5603498, 0.57193005, 0.5926777, 0.4662678, 0.48939455,
                           0.43154383,
                           0.49609298, 0.45263392, 0.4176129, 0.4183942, 0.4502043, 0.5041737, 0.5267499, 0.33614668,
                           0.3420443,
                           0.31359345, 0.47668445, 0.3460606, 0.4751892, 0.33051813, 0.33218288, 0.29892096, 0.3554212,
                           0.33781272,
                           0.2961744, 0.3463757, 0.2773345, 0.4100308, 0.35879332, 0.29014617, 0.49785173, 0.3051463,
                           0.2515564,
                           0.47034898, 0.51030827, 0.7449157, 0.4108481, 0.29660252, 0.37031716, 0.33103842, 0.4949973,
                           0.31136823,
                           0.54252326, 0.34015077, 0.61147845, 0.33463, 0.2696978, 0.38523737, 0.50363505, 0.37288737,
                           0.36658216,
                           0.26108003, 0.29660344, 0.5089753, 0.32892346, 0.37654418, 0.43010494, 0.39048037,
                           0.43493098,
                           0.45357177, 0.366134, 0.47975633, 0.40061533, 0.52896535, 0.3706314, 0.3373877, 0.40647185,
                           0.34383076,
                           0.3458231, 0.3901769, 0.4145405, 0.61124086, 0.3556036, 0.3479371, 0.34148082, 0.45305836,
                           0.33268923,
                           0.4243417, 0.3761385, 0.34342477, 0.37418723, 0.20422915, 0.25409466, 0.3516518, 0.41102523,
                           1.1511544,
                           0.31336966, 0.3708943, 0.38155943, 0.42859265, 0.77132946, 0.38387728, 0.46285582, 0.5644677,
                           0.4211032,
                           0.35043705, 0.25439408],
    'ostatnia warstwa': [1.3373575, 1.136061, 0.752102, 0.7492564, 0.6870458, 0.57367086, 0.59010285, 0.41735506,
                         0.40128684,
                         0.41892496, 0.4513711, 0.35452715, 0.4194119, 0.30355027, 0.43975443, 0.4570434, 0.4253204,
                         0.35971448, 0.45409358, 0.3731226, 0.518126, 0.32418966, 0.44212788, 0.44509763, 0.42627624,
                         0.42851102, 0.45369297, 0.40624657, 0.5237193, 0.48519385, 0.49564806, 0.36307028, 0.53952277,
                         0.3755755, 0.3952077, 0.4204219, 0.39185157, 0.47559932, 0.43906724, 0.36596835, 0.2651781,
                         0.39684886, 0.30584055, 0.7167222, 0.31757903, 0.29646486, 0.47240913, 0.29860747, 0.3268304,
                         0.21555844, 0.30935144, 0.3234672, 0.3616282, 0.7401172, 0.57286066, 0.6160389, 0.31274498,
                         0.35043526, 0.2977443, 0.30573574, 0.37256214, 0.56494695, 0.6057874, 0.41486222, 0.32156932,
                         0.3962555, 0.4067341, 0.5629462, 0.32187933, 0.35275, 0.28968385, 0.43197653, 0.56206304,
                         0.39620286,
                         0.6025871, 0.702065, 0.43837076, 0.31347972, 0.19835526, 0.2854609, 0.34629065, 0.47629544,
                         0.36322576, 0.35651386, 0.4930686, 0.4474684, 0.49711245, 0.5468051, 0.34375134, 0.3259455,
                         0.6785009, 0.42189538, 0.3888277, 0.28127253, 0.30803242, 0.29372627, 0.3493134, 0.48374888,
                         0.32959253, 0.3855428],
    'imputacja': [1.3324138, 0.914784, 0.6798818, 0.50736123, 0.61769617, 0.58329916, 0.51816106, 0.48308372,
                  0.44969591, 0.50139916, 0.40447906, 0.42188376, 0.46694276, 0.52052695, 0.44492832, 0.49211752,
                  0.5585858, 0.60944176, 0.6184076, 0.6551274, 0.49102685, 0.4619101, 0.47231174, 0.5270646,
                  0.44850832, 0.574803, 0.48002812, 0.61568993, 0.36005205, 0.63667446, 0.4935732, 0.4346743,
                  0.5164724, 0.48815528, 0.41317815, 0.50848293, 0.3548589, 0.40769362, 0.3259347, 0.3692541,
                  0.6849407, 0.30419025, 0.3412327, 0.4495486, 0.312051, 0.32954174, 0.52741677, 0.6814225,
                  0.54995394, 0.43062446, 0.55737585, 1.3476856, 0.9865233, 0.39963013, 0.57236624, 0.5201908,
                  0.44770685, 0.46598807, 0.77101624, 0.4017964, 0.3052231, 0.4115246, 0.63078576, 0.9388418,
                  0.39395317, 0.49050906, 0.6004604, 0.43667033, 0.45168465, 0.2839136, 0.5902448, 0.63828075,
                  0.5220462, 0.35818797, 0.39612746, 0.8284782, 0.46471506, 0.34850886, 0.88352895, 0.49356633,
                  0.7844356, 0.50853294, 0.54632545, 0.47543168, 0.35003403, 0.5804066, 0.25348413, 0.5979739,
                  0.8582169, 1.1767347, 0.47916877, 0.43505588, 0.35281867, 0.4659039, 0.54161537, 0.5970218,
                  0.47194162, 0.33900997, 0.98184997, 0.39658734]}

plt.figure(figsize=(9,8))
for key, data in results.items():
    plt.plot([x for x in range(1, 101)], data, label=key)
    plt.xlabel('epoki')
    plt.ylabel('koszt')
plt.legend()
plt.savefig('Training_loss_classifier_fc.png')
