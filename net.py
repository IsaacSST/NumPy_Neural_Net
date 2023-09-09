

import numpy as np
import random


def sig(z):
    return 1.0 / (1.0 + np.exp((-1)*z))


def dsig(z):
    return sig(z) * (1.0 - sig(z))


def cost(output, actual):
    return (np.linalg.norm(output - actual)) ** 2

def cross_entropy(output,actual):
    return (-1)*sum([])


def generate_batches(batch_size, data_size):
    numbatches = data_size // batch_size
    return np.random.choice(range(0, data_size), size=(numbatches, batch_size), replace=False)


class Net(object):
    def __init__(self, layersizes):
        self.nlayers = len(layersizes)
        self.layersizes = layersizes
        self.biases = [np.random.randn(n, 1) for n in self.layersizes[1:]]
        self.weights = [np.random.randn(n, m) for m, n in zip(self.layersizes[:-1], self.layersizes[1:])]

    def output(self, input):
        a = input.flatten().reshape(-1, 1)
        for i in range(0, self.nlayers - 1):
            a = sig(np.matmul(self.weights[i], a) + self.biases[i])

        return a

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
        et = learning_rate
        y = y_train
        X = np.array(list(map(lambda x: x.flatten(), X_train)))
        maxcorrect = 0
        numbatches = X.shape[0] // batch_size
        for j in range(0, epochs):
            batch_indices = generate_batches(batch_size, X.shape[0])
            for i in range(0, batch_indices.shape[0]):
                dCdw, dCdb = self.approximate_gradients(X[batch_indices[i]], y[batch_indices[i]], batch_size)
                self.biases = [b - et*D for b, D in zip(self.biases, dCdb)]
                self.weights = [w - et*D for w, D in zip(self.weights, dCdw)]

                if (i + 1) % 50 == 0:
                   print('\rEpoch ' + str(j+1) + ': Batch ' + str(i+1) + '/' + str(numbatches), end='')

            print("\nFinished Epoch " + str(j+1))
            ncorrect = 0
            for i in range(0, X_test.shape[0]):
                pred = self.predict(X_test[i])
                if pred == np.argmax(y_test[i]):
                    ncorrect += 1

            print('test images correctly guessed is ' + str(ncorrect) + '\n')

            if ncorrect > maxcorrect:
                maxcorrect = ncorrect
            elif ncorrect < maxcorrect - 300:
                print('overfitting break')
                break

    def predict(self, input):
        return np.argmax(self.output(input))

    def backprop(self,sampleX,sampleY):
        acts = [np.random.randn(n, 1) for n in self.layersizes]
        zs = [np.random.randn(n, 1) for n in self.layersizes]
        errors = [np.random.randn(n, 1) for n in self.layersizes[1:]]
        dCdw = [np.random.randn(n, m) for m, n in zip(self.layersizes[:-1], self.layersizes[1:])]

        #feedforward
        acts[0] = np.array(sampleX.reshape(-1, 1))
        zs[0] = np.array(sampleX.reshape(-1, 1))
        for i in range(0, self.nlayers - 1):
            zs[i + 1] = np.matmul(self.weights[i], acts[i]) + self.biases[i]
            acts[i + 1] = sig(zs[i + 1])

        #backpropagation
        #last layer
        dC = acts[-1] - sampleY
        errors[-1] = dC * dsig(zs[-1])
        dCdw[-1] = np.outer(errors[-1], acts[-2])

        #remaining layers
        for i in range(0, self.nlayers - 2):
            errors[-2 - i] = np.matmul(np.transpose(self.weights[-1 - i]), errors[-1 - i]) * dsig(zs[-2 - i])
            dCdw[-2 - i] = np.outer(errors[-2 - i], acts[-3 - i])

        return dCdw, errors

    def approximate_gradients(self, batchX, batchY, batch_size):
        m = batch_size
        dCdw, dCdb = self.backprop(batchX[0], batchY[0])
        for i in range(1, m):
            thisdw, thisdb = self.backprop(batchX[i], batchY[i])
            dCdw = [(d + t) / m for d, t in zip(dCdw, thisdw)]
            dCdb = [(d + t) / m for d, t in zip(dCdb, thisdb)]

        return dCdw, dCdb













