"""算法的评价指标"""

import numpy as np


def rmse(y_pred, y_true):
    """RMSE"""
    t1 = np.sum((y_pred - y_true) ** 2) / np.size(y_true)
    return np.sqrt(t1)


def accuracy(y_pred, y_true):
    acc_list = []
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if a < 0:
            acc_list.append(0)
        elif max(a, b) == 0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)


def nd(y_pred, y_true):
    """Normalized deviation"""
    t1 = np.sum(abs(y_pred - y_true)) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2


def smape(y_pred, y_true):
    s = 0
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a - b) / (abs(a) + abs(b))
    return s / np.size(y_true)


def nrmse(y_pred, y_true):
    """Normalized RMSE"""
    t1 = np.linalg.norm(y_pred - y_true) ** 2 / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2


def metrics(y_pred, y_true):
    return {
        'acc': accuracy(y_pred, y_true),
        'rmse': rmse(y_pred, y_true),
        'nrmse': nrmse(y_pred, y_true),
        'nd': nd(y_pred, y_true),
        'smape': smape(y_pred, y_true)
    }
