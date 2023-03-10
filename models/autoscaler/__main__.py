import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.autoscaler.arima_v2 import BHTARIMA
from models.autoscaler.lstm_v2 import LstmParam, LstmNetwork, ToyLossLayer
from models.utils.dataset import get_one_machine
from models.utils.params import args


def run_lstm(X, y):
    # 正则化数据
    ss = StandardScaler()
    std_X = ss.fit_transform(X)
    std_y = ss.fit_transform(y)

    # (3) 初始化 LSTM 模型
    lstm_param = LstmParam(mem_cell_ct, X.shape[1])
    lstm_net = LstmNetwork(lstm_param)

    # 打印一些有用调试信息
    # print('X.shape = ', X.shape)
    # print('y.shape = ', y.shape)

    # (4) 训练 LSTM 模型
    for epoch in range(5000):
        # print("iter", "%2s" % str(epoch), end=": ")
        for i in range(len(std_y)):
            lstm_net.x_list_add(std_X[i])

        # (5) 预测和计算损失
        # print("y_pred = [" +
        #       ", ".join(["% 2.5f"
        #                  % ss.inverse_transform(lstm_net.lstm_node_list[i].state.h[0].reshape(1, -1))
        #                  for i in range(len(std_y))]) +
        #       "]", end=", ")
        loss = lstm_net.y_list_is(std_y, ToyLossLayer)  # 计算损失
        # print("loss:", "%.3e" % loss)

        # (6) 更新模型
        lstm_param.apply_diff(lr=0.01)
        lstm_net.x_list_clear()  # 清理掉原来的参数

    # (7) 数据后处理：还原 y
    # origin_y = ss.inverse_transform(std_y)
    y_pred = [ss.inverse_transform(lstm_net.lstm_node_list[i].state.h[0].reshape(1, -1)) for i in range(len(std_y))]
    y_hat = [item[0][0] for item in y_pred]

    return y_hat


def run_arima(X, y):
    # 正则化数据
    # ss = MinMaxScaler()  # SVD 默认包含正则化：https://stackoverflow.com/a/46025739/16733647
    # std_X = ss.fit_transform(X)
    # std_y = ss.fit_transform(y)

    # parameters setting
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

    return y_hat


if __name__ == '__main__':

    selected_container_usage_path = args.selected_container_usage_path

    if not os.path.exists(selected_container_usage_path):
        print("container_usage.csv has not been selected.")

    df = pd.read_csv(selected_container_usage_path)
    rows = df.shape[0]

    idx = 0
    while idx < rows:
        # (1) 每次从文件中读取一个 task 的资源需求变化
        machine, idx = get_one_machine(df, idx)

        # training_data_cpu = machine.iloc[:, 3:4].values  # CPU
        # training_data_mem = machine.iloc[:, 4:5].values  # memory
        training_data = machine.iloc[:, 3:5].values  # CPU 和 memory

        # plt.plot(training_data, label="cpu_util_percent")
        # plt.show()

        # (2) 准备数据
        np.random.seed(0)
        mem_cell_ct = 100

        # x_dim = 50
        # y = [-0.5, 0.2, 0.1, -0.5]  # 真实值
        # X = [np.random.random(x_dim) for _ in y]  # shape = [len(y), len(x_dim)]
        # 原始数据
        X = training_data[:-1].T  # shape = [n_samples, n_features]
        y = training_data[-1].reshape(-1, 1)  # shape = [n_samples, n_label_features]
        print('y = \n{}'.format(y))

        # 调用 LSTM 预测模型
        y_hat_lstm = run_lstm(X, y)
        print('y_hat_lstm = {}'.format(y_hat_lstm))
        # print("Pure LSTM loss: \n{}".format(get_index(y_hat_lstm, y)))

        # 调用 ARIMA 预测模型
        y_hat_arima = run_arima(X, y)
        print('y_hat_arima = {}'.format(y_hat_arima))
        # print("Pure ARIMA loss: \n{}".format(get_index(y_hat_arima, y)))

        print('-------------- one epoch is end ----------------')

        break
