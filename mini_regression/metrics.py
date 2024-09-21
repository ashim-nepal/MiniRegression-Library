import numpy as np


def get_mse(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


def get_mae(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))


def get_rmse(y_true, y_pred):

    return np.sqrt(get_mse(y_true, y_pred))


def get_r2(y_true, y_pred):

    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    explained_variance = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (explained_variance / total_variance)
    return r2
