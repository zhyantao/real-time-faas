"""
时序差分 LSTM 实现

Adapt based on: https://github.com/nicodjimenez/lstm
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.utils.dataset import get_one_machine
from models.utils.params import args


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values * (1 - values)


def tanh_derivative(values):
    return 1. - values ** 2


# creates uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct  # 记忆单元的个数 mem cell count
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr=1.0):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)


class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)  # h 的下导数
        self.bottom_diff_s = np.zeros_like(self.s)  # s 的下导数


class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    # 正向更新
    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        # if this is the first lstm node in the network
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        # self.state.h = np.tanh(self.state.s) * self.state.o # version 2
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    # 反向传播
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        # ds = self.state.o * (1 - self.state.s ** 2) * top_diff_h + top_diff_s  # version 1
        # ds = self.state.o * top_diff_h + np.tanh(top_diff_s) # version 2
        ds = self.state.o * top_diff_h + top_diff_s
        # do = np.tanh(self.state.s) * top_diff_h # version 2
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        # we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """

    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def sliding_windows(data, seq_length):
    X = []
    y = []

    for i in range(data.shape[1] - seq_length - 1):
        _X = data[:, i:i + seq_length]  # 每次截取一段
        _y = data[:, i + seq_length]  # 每次拼接一个
        X.append(_X)
        y.append(_y)

    return np.array(X), np.array(y)


def run_lstm_multi_step(X, y):
    # 正则化数据
    ss = StandardScaler()
    std_data = ss.fit_transform(X)
    # std_y = ss.fit_transform(y)

    seq_len = 4

    sw_X, sw_y = sliding_windows(std_data, seq_len)
    # print(sw_X, sw_y)

    # 划分数据集
    train_size = int(sw_X.shape[0] * 0.67)
    test_size = sw_X.shape[0] - train_size

    # data_X = np.array(std_data)
    # data_y = np.array(std_data)

    train_X = np.array(sw_X[:train_size])
    train_y = np.array(sw_y[:train_size])

    test_X = np.array(sw_X[train_size:])
    test_y = np.array(sw_y[train_size:])

    # (1) 初始化 LSTM 模型
    lstm_param = LstmParam(mem_cell_ct=100, x_dim=seq_len)
    lstm_net = LstmNetwork(lstm_param)

    # 打印一些有用调试信息
    # print('X.shape = ', X.shape)
    # print('y.shape = ', y.shape)

    # (2) 训练 LSTM 模型
    for epoch in range(100):
        # print("iter", "%2s" % str(epoch), end=": ")
        for layer in range(train_X.shape[1]):
            for i in range(train_X.shape[0]):
                lstm_net.x_list_add(train_X[i, layer])

            # for i in range(train_X.shape[1]):
            #     lstm_net.x_list_add(train_X[:, i])
            # for i in range(train_X.shape[0]):
            #     lstm_net.x_list_add(train_X[i])

            # (3) 预测和计算损失
            print("y_pred = [" +
                  ", ".join(["% 2.5f"
                             # % ss.inverse_transform(lstm_net.lstm_node_list[m].state.h[0].reshape(1, -1))
                             % lstm_net.lstm_node_list[m].state.h[0]
                             for m in range(train_y.shape[0])]) +
                  "]", end=", ")
            loss = lstm_net.y_list_is(train_y[:, 0], ToyLossLayer)  # 计算损失
            print("loss:", "%.3e" % loss)

            # (4) 更新模型
            lstm_param.apply_diff(lr=0.01)
            lstm_net.x_list_clear()  # 清理掉原来的参数

    # (5) 预测 y
    # origin_y = ss.inverse_transform(std_y)
    y_hat = (np.zeros_like(test_y)).reshape(-1)
    for layer in range(test_X.shape[1]):
        for i in range(test_X.shape[0]):
            lstm_net.x_list_add(test_X[i, layer])
    for m in range(train_y.shape[0]):
        pred = lstm_net.lstm_node_list[m].state.h[0]
        # pred = ss.inverse_transform(lstm_net.lstm_node_list[i].state.h[0].reshape(1, -1))
        # y_hat[m] = ss.inverse_transform(pred.reshape(1, -1))
        y_hat[m] = pred
    print('-----> test_y: ', test_y)
    print('-----> y_hat: ', y_hat)
    loss = lstm_net.y_list_is(train_y[:, 0], ToyLossLayer)  # 计算损失
    print("loss:", "%.3e" % loss)
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

        X = training_data.T
        y = training_data[-1].reshape(-1, 1)

        run_lstm_multi_step(X, y)

        break
