import matplotlib.pyplot as plt
import os


class Visualizator:
    def __init__(self, save_dir, file_name, nn ):
        self.save_dir = save_dir
        self.file_name = file_name
        self.nn = nn

    def plot_loss(self, loss, epochs, title):
        plt.plot(epochs, loss)
        plt.title(title)
        plt.savefig(os.path.join(self.save_dir, self.file_name))
        plt.close()

    def draw_mnist_image(self, i, j, g, ):
        _, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.imshow(g[j].reshape([28, 28]), origin="upper", cmap="gray")
        ax.axis('off')
        plt.savefig(os.path.join(self.save_dir, "".join(
            (str(i * self.nn + j), "-last_layer.png"))),
                    bbox_inches='tight')
        plt.close()
