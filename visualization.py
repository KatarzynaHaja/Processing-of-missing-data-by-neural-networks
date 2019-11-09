import matplotlib.pyplot as plt
import os
import pathlib


class Visualizator:
    def __init__(self, save_dir, file_name, nn):
        self.save_dir = save_dir
        self.file_name = file_name
        self.nn = nn

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_loss(self, loss, epochs, title, file_name):
        plt.plot(epochs, loss)
        plt.title(title)
        plt.savefig(os.path.join(self.save_dir, file_name))
        plt.close()

    def draw_mnist_image(self, i, j, g, method):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(g[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join(self.save_dir, "".join(
            (str(i * self.nn + j), '-' + method + '.png'))),
                    bbox_inches='tight')
        plt.close()

    def draw_svhn_image(self, i, j, g, method):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(g[j].reshape([32, 32, 3]), origin="upper")
        ax.axis('off')
        plt.savefig(os.path.join(self.save_dir, "".join(
            (str(i * self.nn + j), '-' + method + '.png'))),
                    bbox_inches='tight')
        plt.close()



# import sys
# from PIL import Image
#
# for i in range(1000):
#     images = [Image.open(x) for x in ['original_data/'+str(i)+'.png',
#                                       'image_with_patch/'+str(i)+'.png',
#                                       'result_different_cost_250_10_2.0/'+str(i)+'-different_cost.png',
#                                       'result_first_layer_150_10_1.0/'+str(i)+'-first_layer.png',
#                                       'result_last_layer_150_10_0.0/'+str(i)+'-last_layer.png']
#               ]
#     widths, heights = zip(*(i.size for i in images))
#
#     total_width = sum(widths)
#     max_height = max(heights)
#
#     new_im = Image.new('RGB', (total_width, max_height))
#
#     x_offset = 0
#     for im in images:
#       new_im.paste(im, (x_offset,0))
#       x_offset += im.size[0]
#
#     new_im.save('my_methods/my_methods-'+str(i)+'.png')
#
#
# for i in range(1000):
#     images = [Image.open(x) for x in ['original_data/'+str(i)+'.png',
#                                       'image_with_patch/'+str(i)+'.png',
#                                       'result_imputation_250_1_0.0/' + str(i) + '-imputation.png',
#                                       'result_theirs_250_1_0.0/'+str(i)+'-theirs.png',
#                                       'result_last_layer_150_20_0.0/'+str(i)+'-last_layer.png'
#                                       ]
#               ]
#     widths, heights = zip(*(i.size for i in images))
#
#     total_width = sum(widths)
#     max_height = max(heights)
#
#     new_im = Image.new('RGB', (total_width, max_height))
#
#     x_offset = 0
#     for im in images:
#       new_im.paste(im, (x_offset,0))
#       x_offset += im.size[0]
#
#     new_im.save('comparison/comparison-'+str(i)+'.png')
#
#
# images = [Image.open(x) for x in ['my_method_comparison/my_methods-0.png',
#                                   'my_method_comparison/my_methods-2.png',
#                                   'my_method_comparison/my_methods-4.png',
#                                   'my_method_comparison/my_methods-16.png',
#                                   'my_method_comparison/my_methods-30.png'
#                                   ]
#           ]
# widths, heights = zip(*(i.size for i in images))
#
# max_width = max(widths)
# total_height = sum(heights)
#
# new_im = Image.new('RGB', (max_width, total_height))
#
# y_offset = 0
# for im in images:
#     new_im.paste(im, (0, y_offset))
#     y_offset += im.size[1]
#
# new_im.save('my_method_comparison/comparison_all.png')
