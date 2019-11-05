import numpy as np
import matplotlib.pyplot as plt

data_folder = "../data/cifar-10-batches-py"


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_data():
    """
    data : (10 000 x 3072)
    labels : (10 000 x 1)
    one_hot : (10 000 x 10)
    """
    filename = f"{data_folder}/data_batch_1"
    file = unpickle(filename)

    labels = file[b"labels"]
    data = file[b"data"]
    no_classes = 10
    N = len(labels)

    one_hot = np.zeros((N, no_classes))
    one_hot[np.arange(N), labels] = 1

    labels = np.array(labels).reshape(-1, 1)

    # normalize
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)

    X = (data - mean) / std
    return X, labels, one_hot, mean, std


def ReLU(x):
    return np.where(x > 0, x, 0)


def dReLU(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def plot_image(x, mean, std):
    x = (x.T * std + mean).astype(int)
    img = x.reshape(3, 32, 32)
    plt.figure(figsize=(2, 2))
    plt.imshow(np.transpose(img, (1, 2, 0)))


def check_if_correct(y, p):
    temp = np.argmax(y, axis=1) - np.argmax(p.T, axis=1)
    correct_ones = np.where(temp == 0, 1, 0)
    return np.sum(correct_ones)


class Plotter:
    def __init__(self, title):
        self.title = title
        self.x = []
        self.y = []

    def add(self, epoch, cost):
        self.y.append(cost)
        self.x.append(epoch)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)

        ax.set(xlabel="epochs", ylabel="cost", title=self.title)
        ax.grid()
        # fig.savefig("{}.png".format(self.title))
        plt.show()
