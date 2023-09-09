
import numpy as np
import gzip


class Reader(object):
    def __init__(self, directory=''):
        self.directory = directory

    def read_data(self, normalize=True, vectorize_label=False):
        X_train = read_images(self.directory + '/train-images-idx3-ubyte.gz')
        y_train = read_labels(self.directory + '/train-labels-idx1-ubyte.gz')
        X_test = read_images(self.directory + '/t10k-images-idx3-ubyte.gz')
        y_test = read_labels(self.directory + '/t10k-labels-idx1-ubyte.gz')
        if normalize:
            X_train = X_train / 255
            X_test = X_test / 255

        if vectorize_label:
            y_train = np.array(list(map(label_to_vec, y_train)))
            y_test = np.array(list(map(label_to_vec, y_test)))

        return X_train, y_train, X_test, y_test


def read_images(file):
    with gzip.open(file, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        nimages = int.from_bytes(f.read(4), 'big')
        imagerows = int.from_bytes(f.read(4), 'big')
        imagecols = int.from_bytes(f.read(4), 'big')
        data = f.read()

    return np.frombuffer(data, dtype=np.uint8).reshape((nimages, imagerows, imagecols))


def read_labels(file):
    with gzip.open(file, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        nlabels = int.from_bytes(f.read(4), 'big')
        data = f.read()

    return np.frombuffer(data, dtype=np.uint8)


def label_to_vec(label):
    vec = np.zeros((10, 1))
    vec[label] = 1.0
    return vec
