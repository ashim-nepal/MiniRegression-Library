import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle = True, random_seed = None):
    X= np.array(X)
    y = np.array(y)

    if random_seed is not None:
        np.random.seed(random_seed)

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    split_index = int(X.shape[0] * (1 - test_size))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test
