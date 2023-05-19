import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from models.scheduler.gcn import accuracy, GCN


def test_gcn_with_random_numbers():
    """
    使用随机生成的数字来测试 GCN 算法是否能正常工作
    :return:
    """
    # 初始化状态空间
    adj_matrix = torch.randn(100, 100)  # 任务间的依赖信息
    n_samples = adj_matrix.shape[0]
    features = torch.randn(n_samples, 4)  # DAG 中每个节点的 feature
    labels = torch.rand(n_samples).long()  # DAG 中每个节点的 label，一维张量

    # 初始化超参数
    train_size = 0.8
    n_train_samples = np.ceil(n_samples * train_size)
    idx_train = np.arange(n_train_samples)
    idx_test = np.arange(n_train_samples, n_samples)

    # 初始化网络模型
    model = GCN(features.shape[1], 8, int(labels.max()) + 1, 0.6)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 开始训练
    start_time = time.time()
    for epoch in range(1000):
        model.train()  # 计算梯度并使用反向传播算法来更新权重
        optimizer.zero_grad()  # 将模型参数的梯度清零

        # 计算训练集上的损失
        output = model(features, adj_matrix)  # 预测
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # 计算评估集上的损失
        loss_val = F.nll_loss(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_test], labels[idx_test])

        # 打印一些有用的信息
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - start_time))
