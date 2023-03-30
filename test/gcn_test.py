import time
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models.scheduler.mine.gcn import GCN


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GCNTest(unittest.TestCase):

    def test_gcn(self):
        # 初始化状态空间
        adj_matrix = torch.randn(100, 100)  # DAG
        n_samples = adj_matrix.shape[0]
        features = torch.randn(n_samples, 20)  # DAG 中每个节点的 feature
        labels = torch.rand(n_samples).long()  # DAG 中每个节点的 label，一维张量

        # 初始化超参数
        train_size = 0.8
        n_train_samples = np.ceil(n_samples * train_size)
        idx_train = np.arange(n_train_samples)
        idx_test = np.arange(n_train_samples, n_samples)

        # 初始化网络模型
        model = GCN(nfeat=features.shape[1],  # 列数
                    nhid=8,  # 隐藏层的个数
                    nclass=int(labels.max()) + 1,  # 类别数
                    dropout=0.6)

        # 设置优化器
        optimizer = optim.Adam(model.parameters(),
                               lr=0.005,
                               weight_decay=5e-4)

        t = time.time()

        epochs = 1000
        for epoch in range(epochs):
            # 训练和更新参数
            model.train()
            optimizer.zero_grad()  # 更新参数
            output = model(features, adj_matrix)  # 预测结果
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 计算像 -log 一样的损失
            loss_train.backward()
            optimizer.step()
            acc_train = accuracy(output[idx_train], labels[idx_train])

            # 检查模型正确率
            loss_val = F.nll_loss(output[idx_test], labels[idx_test])
            acc_val = accuracy(output[idx_test], labels[idx_test])

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))
