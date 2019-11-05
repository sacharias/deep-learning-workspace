import numpy as np
import time
from utils import load_data, ReLU, dReLU, softmax, check_if_correct

lr = 0.05
batch_size = 100
w_decay = 0


class Layer:
    def __init__(self, in_size, out_size, batch_norm):
        self.W = np.random.rand(out_size, in_size) * 0.0001
        self.b = np.zeros((out_size, 1))
        self.batch_norm = batch_norm

        if batch_norm:
            self.gamma = np.random.rand(out_size, 1)
            self.beta = np.zeros((out_size, 1))
    
    def forward(self, X):
        S = np.dot(self.W, X) + self.b

        if self.batch_norm:
            self.mu = np.sum(S, axis=1).reshape(-1, 1) / batch_size
            self.var = np.var(S, axis=1)

            S_hat = np.dot(np.diag( (self.var + 1e-15) ** (-0.5) ), (S - self.mu))
            S_t = self.gamma * S_hat + self.beta
            X1 = ReLU(S_t)
            return X1

        else:
            X1 = ReLU(S)
            return X1



class ProbLayer:
    def __init__(self, in_size, out_size):
        self.W = np.random.rand(out_size, in_size) * 0.0001
        self.b = np.zeros((out_size, 1))

    def forward(self, X):
        S = np.dot(self.W, X) + self.b
        X = softmax(S)
        return X


class Network:
    layers = []
    batch_norm = True

    def __init__(self, layer_sizes):
        """First value is input size, last value is output size"""
        n = len(layer_sizes) - 1

        for i in range(n):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            if i == n - 1:
                l = ProbLayer(in_size, out_size)
                self.layers.append(l)
            else:
                l = Layer(in_size, out_size, self.batch_norm)
                self.layers.append(l)

    def train(self, X0, onehot):
        # forward
        X = X0
        for layer in self.layers:
            X = layer.forward(X)
        
        # Log Loss
        L2 = 0
        for layer in self.layers:
            L2 += np.sum(layer.W ** 2)
        
        loss = -np.sum(onehot * np.log(1e-15 + X.T)) + w_decay * L2
        loss = loss / batch_size
        accuracy = check_if_correct(onehot, X)

        # Backprop
        G = X - onehot.T

        if self.batch_norm:
            # fortsätt här
        

        print(loss)

def main():
    data_train, labels_train, onehots_train, mean_train, std_train = load_data()
    
    network = Network([3072, 50, 10])

    iterations = data_train.shape[0] // batch_size
    start_t = time.time()

    for epoch in range(2):
        avg_loss = 0
        accuracy = 0

        for idx in range(iterations):
            start = batch_size * idx
            end = batch_size * (idx + 1)

            X0 = data_train[start:end,:].T # 3072x100
            onehot = onehots_train[start:end,:] # 100x10

            network.train(X0, onehot)

            break





main()
print("\nEnd\n")
