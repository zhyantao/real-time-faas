"""
读取某个作业和它的需求
"""
from __future__ import division
from __future__ import print_function

import glob
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.scheduler.gcn import load_data_from_job, GCN, accuracy
from models.utils.dataset import get_one_job, para

if __name__ == '__main__':
    df = pd.read_csv(para.get("selected_batch_task_path"))

    rows = df.shape[0]  # CSV 文件的行数
    idx = 0
    while idx < rows:
        job, idx = get_one_job(df, idx)
        print(job)

        adj, features, labels, idx_train, idx_val, idx_test = load_data_from_job(job)  # 每次分析一个 job

        ############################################
        # 构建网络，指定优化器
        ############################################
        model = GCN(nfeat=features.shape[1],  # 列数
                    nhid=8,  # 隐藏层的个数
                    nclass=int(labels.max()) + 1,  # 类别数
                    dropout=0.6)  # 随机丢失，防止过拟合

        optimizer = optim.Adam(model.parameters(),
                               lr=0.005,
                               weight_decay=5e-4)

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)


        ############################################
        # 训练网络
        ############################################
        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            model.eval()  # 每次训练都会评估效果
            output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            return loss_val.data.item()


        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = 1000 + 1  # epochs + 1
        best_epoch = 0
        for epoch in range(1000):
            loss_values.append(train(epoch))

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == 100:  # 早停策略
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        ############################################
        # 测试网络的预测效果
        ############################################
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))

        break
