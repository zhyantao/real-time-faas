"""Adapted according to https://github.com/nicodjimenez/lstm"""
import os

import numpy as np
import pandas as pd

from models.utils.parameters import args
from models.utils.tools import get_one_machine


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
        """初始化 LSTM 参数"""
        self.mem_cell_ct = mem_cell_ct  # 记忆单元的个数 mem cell count
        self.x_dim = x_dim  # 输入 x 的维度，表示需要有多少个先验数据用来学习
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
        """更新参数"""
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
        self.s_prev = None
        self.h_prev = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        """正向更新"""
        # if this is the first lstm node in the network
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x, h_prev))  # 水平按列堆叠数组
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg + self.state.bottom_diff_s)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi + self.state.bottom_diff_s)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        """误差反向传播"""
        ds = self.state.o * top_diff_h + top_diff_s  # dL(t) / ds(t) = .... 注意 s(t) 本身是一个递归函数, 需要特殊处理
        do = self.state.s * top_diff_h  # dL(t) / do(t) = .... 因为将 o(t) 看做变量时, s(t) 可看做常数处理
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # vector inside sigma / tanh function 的导数
        di_input = sigmoid_derivative(self.state.i) * di  # 求误差对输入门的输入的导数
        df_input = sigmoid_derivative(self.state.f) * df  # 求误差对遗忘门的输入的导数
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # inputs 的导数
        self.param.wi_diff += np.outer(di_input, self.xc)  # 求误差对输入门上的参数的导数
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # 计算 bottom diff
        # 计算记忆单元输入的累积误差，误差来源于所有的输入的误差累积和
        dxc = np.zeros_like(self.xc)  # dxc 表示记忆单元的输入 diff of x_{cell}
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # 保存 bottom diffs
        self.state.bottom_diff_s = ds * self.state.f  # dL(t) / ds(t)
        self.state.bottom_diff_h = dxc[self.param.x_dim:]  # dL(t) / dh(t-1)


class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_truth, loss_layer):
        """
        根据真实值和预测值计算损失。
        注意，该函数并不会更新模型参数，要更新模型参数，需要调用 self.lstm_param.apply_diff()。
        :param y_truth: 真实值
        :param loss_layer: 损失函数
        :return:
        """
        assert len(y_truth) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        # 从  T 到 1 反向计算损失
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_truth[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_truth[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        # we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_truth[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_truth[idx])
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
    """计算隐藏层数组中第一个元素的平方误差"""

    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


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

        # run_lstm_multi_step(X, y)

        break
