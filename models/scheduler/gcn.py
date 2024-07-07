"""
Adapted according to https://github.com/tkipf/pygcn
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import Parameter

from models.utils.dataset import gen_gcn_dqn_input


class GraphConvolution(nn.Module):
    """
    定义一个图卷积单元的计算方式，在 GCN 类中可以组合多个单元。
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()  # 初始化父类对象，后面可直接使用父类方法：forward、backward 等
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # (in_features, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # 根据 out_features 取标准差
        self.weight.data.uniform_(-stdv, stdv)  # 将 weight 置为符合 uniform 分布的变量，其值介于 [-stdv, stdv] 之间
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 矩阵乘法
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()

        # 定义一个含有两层卷积图卷积网络
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def accuracy(output, labels):
    """
    计算预测结果的正确率
    :param output:
    :param labels:
    :return:
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def encoding(adj_matrix, features):
    # 初始化超参数
    n_samples = features.shape[0]
    in_features = features.shape[1]
    n_hidden = 8
    out_features = n_samples  # 将 GCN 的输出特征的个数设置为与 n_samples 一样，是希望每个 sample 都能有自己的特点
    dropout_rate = 0.6

    # 划分训练集和测试集
    train_size = 0.5
    n_train_samples = np.ceil(n_samples * train_size)
    idx_train = np.arange(n_train_samples)
    idx_test = np.arange(n_train_samples, n_samples)

    labels = torch.LongTensor(range(n_samples))  # DAG 中每个节点的 label，一维张量

    # 初始化网络模型
    model = GCN(in_features, n_hidden, out_features, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 开始训练
    start_time = time.time()
    max_acc = -1
    best_output = None
    for epoch in range(5000):
        model.train()  # 计算梯度并使用反向传播算法来更新权重
        optimizer.zero_grad()  # 将模型参数的梯度清零

        # 计算训练集上的损失
        output = model(features, adj_matrix)  # 调用 forward 函数，计算每个 sample 的概率分布
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 计算对数损失
        loss_train.backward()
        optimizer.step()
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # 计算评估集上的损失
        loss_val = F.nll_loss(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_test], labels[idx_test])

        if acc_train > max_acc:
            max_acc = acc_train
            best_output = output

        # 打印一些有用的信息
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - start_time))

    return best_output


if __name__ == '__main__':
    # 初始化状态空间
    task_adj_matrix, task_features, node_adj_matrix, node_features = gen_gcn_dqn_input()
    task_adj_matrix = torch.Tensor(task_adj_matrix)
    task_features = torch.Tensor(task_features)  # (n_samples, n_features)
    node_adj_matrix = torch.Tensor(node_adj_matrix)
    node_features = torch.Tensor(node_features)

    # features = np.concatenate((task_adj_matrix, task_features), axis=1)  # task 的特征拼接

    encoded_tasks = encoding(task_adj_matrix, task_features)
    encoded_nodes = encoding(node_adj_matrix, node_features)

    print('encoded_tasks.shape: ', encoded_tasks.shape)
    print('encoded_nodes.shape: ', encoded_nodes.shape)
