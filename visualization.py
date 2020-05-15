import matplotlib.pyplot as plt
import os
import pathlib


class Visualizator:
    def __init__(self, save_dir, file_name, nn=None):
        self.save_dir = save_dir
        self.file_name = file_name
        self.nn = nn

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_train_multiple_loss(self, losses, epochs, title, file_name):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for loss in losses:
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

    @classmethod
    def draw_losses(self, losses, title, legend):
        cmap = plt.get_cmap(name='hsv')
        for i in range(losses):
            plt.title(title, font=14)
            plt.plot([j for j in range(len(losses[i]))],losses[i],  color=cmap(i), label=legend[i])
        plt.legend()
        plt.savefig(title)





