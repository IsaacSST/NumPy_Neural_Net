#for testing. neural net code in net.py

import numpy as np
from PIL import Image
import net
import MNISTreader
from os.path import dirname, abspath

def main():
    MNIST = MNISTreader.Reader(dirname(dirname(abspath(__file__))) + '/data/MNIST')
    X_train, y_train, X_test, y_test = MNIST.read_data(normalize=True, vectorize_label=True)
    model = net.Net([784, 28, 10])
    model.train(X_train, y_train, X_test, y_test, 1, 5, 4.0)
    #saving model
    with open(dirname(abspath(__file__)) + '/models/model.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        f.write(str(model.weights))
        f.write('\n\n')
        f.write(str(model.biases))


def display_image(arr):
    Image.fromarray(arr, 'L').show()


def load_model_from_text(file):
    with open(file, 'r') as f:
        data = f.read().split('\n\n')
    weights = eval(data[0].replace('\n        ', ' ').replace('\n       ', ' '))
    biases = eval(data[1].replace('\n       ', ' '))
    layersizes = [weights[0].shape[1]]
    for vec in biases:
        layersizes.append(vec.shape[0])

    return net.Net(layersizes)

if __name__ == "__main__":
    main()