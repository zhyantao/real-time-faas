#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script is the demo of BHT-ARIMA algorithm
# References : "Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting"
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from BHT_ARIMA import BHTARIMA
from BHT_ARIMA.util.utility import get_index
from models.utils.dataset import get_one_machine
from models.utils.params import args

if __name__ == "__main__":
    # prepare data
    # the data should be arranged as (ITEM, TIME) pattern
    # import traffic dataset
    ori_ts = np.load('input/traffic_40.npy').T
    print("shape of data: {}".format(ori_ts.shape))
    print("This dataset have {} series, and each series have {} time step".format(
        ori_ts.shape[0], ori_ts.shape[1]
    ))

    # selected_container_usage_path = args.selected_container_usage_path
    #
    # if not os.path.exists(selected_container_usage_path):
    #     print("container_usage.csv has not been selected.")
    #
    # df = pd.read_csv(selected_container_usage_path)
    # rows = df.shape[0]
    #
    # # f_loss_path = 'loss.csv'
    # # f_loss = pd.DataFrame()  # 创建 loss 文件，用于记录 loss
    # # f_loss.to_csv(f_loss_path, index=False)
    #
    # idx = 0
    # while idx < rows:
    #     # (1) 每次从文件中读取一个 task 的资源需求变化
    #     machine, idx = get_one_machine(df, idx)
    #     # training_data_cpu = machine.iloc[:, 3:4].values  # CPU
    #     # training_data_mem = machine.iloc[:, 4:5].values  # memory
    #     training_data = machine.iloc[:, 3:5].values  # CPU 和 memory
    #     # plt.plot(training_data, label="cpu_util_percent")
    #     # plt.show()
    #
    #     # (2) 准备数据
    #     np.random.seed(0)
    #     mem_cell_ct = 100
    #
    #     # x_dim = 50
    #     # y = [-0.5, 0.2, 0.1, -0.5]  # 真实值
    #     # X = [np.random.random(x_dim) for _ in y]  # shape = [len(y), len(x_dim)]
    #     # 原始数据
    #     X = training_data[:-1].T  # shape = [n_samples, n_features]
    #     x_dim = X.shape[1]
    #     y = training_data[-1].reshape(-1, 1)  # shape = [n_samples, n_label_features]
    #
    #     # 正则化数据
    #     ss = StandardScaler()
    #     std_X = ss.fit_transform(X)
    #     std_y = ss.fit_transform(y)
    #
    #     X = std_X
    #     y = std_y

    # parameters setting
    X = ori_ts[..., :-1]  # training data
    y = ori_ts[..., -1]  # label, take the last time step as label
    n_samples = X.shape[0]

    print('y = ', y)
    p = 3  # p-order
    d = 2  # d-order
    q = 1  # q-order
    taus = [n_samples, 5]  # MDT-rank
    Rs = [5, 5]  # tucker decomposition ranks
    epochs = 10  # iterations
    tol = 0.001  # stop criterion
    Us_mode = 4  # orthogonality mode

    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = BHTARIMA(X, p, d, q, taus, Rs, epochs, tol, verbose=0, Us_mode=Us_mode)
    pred, _ = model.run()
    y_hat = pred[..., -1]
    print('y_hat = ', y_hat)

    # print extracted forecasting result and evaluation indexes
    # print("forecast result(first 10 series):\n", y_hat[:10])

    print("Evaluation index: \n{}".format(get_index(y_hat, y)))
