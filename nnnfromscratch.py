import numpy as np
import tensorflow as tf


(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_full = X_train_full.reshape(X_train_full.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

split = int(0.8 * X_train_full.shape[0])
X_train, X_val = X_train_full[:split], X_train_full[split:]
y_train, y_val = y_train_full[:split], y_train_full[split:]


class ReLU:
    def forward(self, x):
        self.x = x
        self.out = np.maximum(0, x)

    def backward(self, grad):
        self.dx = grad.copy()
        self.dx[self.x <= 0] = 0


class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, grad):
        self.dx = np.zeros_like(grad)
        for i, (o, g) in enumerate(zip(self.out, grad)):
            o = o.reshape(-1, 1)
            jacobian = np.diagflat(o) - o @ o.T
            self.dx[i] = jacobian @ g


class Layer:
    def __init__(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(-limit, limit, (n_in, n_out))
        self.b = np.zeros((1, n_out))

    def forward(self, x):
        self.x = x
        self.out = x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        self.dx = grad @ self.W.T


class CategoricalCrossEntropy:
    def __init__(self, num_classes, smoothing=0.0):
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, y_pred, y_true):




        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if y_true.ndim == 1:
            return -np.log(y_pred[np.arange(len(y_pred)), y_true])
        
        if self.smoothing > 0:
            y_true = (1 - self.smoothing) * y_true + self.smoothing / y_true.shape[1]
            
        return -np.sum(y_true * np.log(y_pred), axis=1)

    def backward(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true = np.eye(self.num_classes)[y_true]
        self.dx = -y_true / (y_pred + 1e-7)
        self.dx = self.dx / y_pred.shape[0]


class NeuralNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, lr):
        self.fc1 = Layer(in_dim, hidden_dim)
        self.act = ReLU()
        self.fc2 = Layer(hidden_dim, out_dim)
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy(out_dim, smoothing=0.1)
        self.lr = lr

    def forward(self, x):
        self.fc1.forward(x)
        self.act.forward(self.fc1.out)
        self.fc2.forward(self.act.out)
        self.softmax.forward(self.fc2.out)
        return self.softmax.out

    def backward(self, y_true):
        self.loss.backward(self.softmax.out, y_true)
        self.softmax.backward(self.loss.dx)
        self.fc2.backward(self.softmax.dx)
        self.act.backward(self.fc2.dx)
        self.fc1.backward(self.act.dx)

        for layer in (self.fc1, self.fc2):
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

    def train(self, X, y, epochs, batch_size):
        for _ in range(epochs):
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]

            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]


                
                self.forward(xb)
                self.backward(yb)

    def evaluate(self, X, y):
        preds = np.argmax(self.forward(X), axis=1)
        return np.mean(preds == y)


net = NeuralNetwork(
    in_dim=X_train.shape[1],
    hidden_dim=128,
    out_dim=10,
    lr=0.01
)

net.train(X_train, y_train, epochs=10, batch_size=32)

print(f"Validation Accuracy: {net.evaluate(X_val, y_val) * 100:.2f}%")
print(f"Test Accuracy: {net.evaluate(X_test, y_test) * 100:.2f}%")
