import numpy as np
import time
from utils import Plotter, load_data, ReLU

np.random.seed(40)

def train():
    plotter = Plotter("batch-norm-tester-".format(time.time()))    

    data, labels, one_hots, mean, std = load_data()
    data_size = data.shape[0]
    lr = 0.05
    w_decay = 0
    no_hidden = 50

    W1, W2 = np.random.rand(no_hidden, 3072) * 0.0001, np.random.rand(10, no_hidden) * 0.0001
    b1, b2 = np.zeros((no_hidden, 1)), np.zeros((10, 1))

    gamma1, beta1 = np.random.rand(no_hidden, 1), np.zeros((no_hidden, 1))

    batch_size = 100
    iterations = data_size // batch_size
    start_t = time.time()

    for epoch in range(400):
        avg_loss = 0
        accuracy = 0
        
        for idx in range(iterations):
            start = batch_size * idx
            end = batch_size * (idx + 1)

            X0 = data[start:end,:].T # 3072x100
            one_hot = one_hots[start:end,:] # 100x10
            
            S1 = np.dot(W1, X0) + b1
            mu_1 = np.sum(S1, axis=1).reshape(-1,1) / batch_size
            var_1 = np.var(S1, axis=1)
            
            S1_hat = np.dot(np.diag( (var_1 + 1e-15) ** (-0.5) ), (S1 - mu_1)) # batchNorm
            S1_t = gamma1 * S1_hat + beta1
            
            X1 = ReLU(S1_t)
            
            S2 = np.dot(W2, X1) + b2
            X2 = softmax(S2) 
            
            # log loss
            L2 = np.sum(W1 ** 2) + np.sum(W2 ** 2)
            loss = -np.sum(one_hot * np.log(1e-15 + X2.T)) + w_decay * L2

            avg_loss += (loss / batch_size)
            accuracy += check_if_correct(one_hot, X2)
            
            # backward
            G = X2 - one_hot.T
            
            dW2 = np.dot(G, X1.T) / batch_size
            dW2 += 2 * w_decay * W2
            db2 = np.sum(G, axis=1).reshape(-1,1) / batch_size
            
            G = np.dot(W2.T, G)
            G = G * dReLU(X1)
            
            dgamma1 = np.sum(G * S1_hat, axis=1, keepdims=True) / batch_size
            dbeta1 = np.sum(G, axis=1, keepdims=True) / batch_size
            
            G = G * gamma1
            
            # == BatchNormBackPass ==
            sigma = var_1.reshape(-1,1) + 1e-15
            sigma1 = sigma ** -0.5
            sigma2 = sigma ** -1.5
            
            G1 = G * sigma1
            G2 = G * sigma2
            
            D = S1 - mu_1
            c = np.sum(G2 * D,axis=1, keepdims=True) # 50x1 osäker
            
            part1 = (1 / batch_size) * (G1 @ np.ones((batch_size, 1))) @ np.ones((1, batch_size))
            part2 = (1 / batch_size) * D * (c @ np.ones((1, batch_size)))
            
            G = G1 - part1 - part2
            # ==========================
            
            dW1 = np.dot(G, X0.T) / batch_size
            dW1 += w_decay * 2 * W1
            
            db1 = np.sum(G, axis=1).reshape(-1,1) / batch_size
            
            # update
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            
            gamma1 = gamma1 - lr * dgamma1
            beta1 = beta1 - lr * dbeta1

        avg_loss /= iterations
        accuracy /= iterations
    
        if (epoch % 10 == 0):
            plotter.add(epoch, avg_loss)
            print("epoch: {} \tloss: {:.3} \tacc: {:.3}".format(epoch, avg_loss, accuracy))

    print("{:.3} s".format(time.time() - start_t))
    plotter.plot()
    
    print(b1, "\n", b2)
    print("done")
train()