# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     script
   Description :
   Author :       Liang Rong
   date：          2022/12/14
-------------------------------------------------
   Change Activity:
                   2022/12/14:
-------------------------------------------------
"""

import pprint

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def encode_onehot(all_labels):
    # 对所有的label 进行编号，再将编号转换成 one_hot向量
    print(all_labels.tolist())
    classes = sorted(list(set(all_labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    pprint.pprint(classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get, all_labels)), dtype=np.int32)
    return labels_onehot


path = "../../../dataset/cora/"
dataset = "cora"
content = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
labels = encode_onehot(content[:, -1])

print(np.identity(7))

print(labels)

print("cora content shape = ", content.shape)

print("features  = ", features.shape)
# print("features[0] = ", features[0])
print("labels shape = ", labels.shape)

idx = np.array(content[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
raw_edges = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
print("raw_edges = ", raw_edges)

edges = np.array(list(map(idx_map.get, raw_edges.flatten())), dtype=np.int32).reshape(raw_edges.shape)
print(" edges = ", edges)

adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
print("adj = ", adj)

test_sp = np.array([[0, 1, 0], [1, 0, 1]])
print(test_sp)
print("sp.csr_matrix = ")
print(sp.csr_matrix(test_sp))
print("sp.csc_matrix = ")
print(sp.csc_matrix(test_sp))

edges = np.array([[0, 1], [1, 2]])

adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(3, 3),
                    dtype=np.float32)
print("adj = ")
print(adj)
print(adj.toarray())

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
print(" adg === ")
print(adj)
print(adj.toarray())

#
# def normalize_adj(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     print("row sum = ", rowsum)
#     r_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     print("r_inv_sqrt = ",r_inv_sqrt)
#     r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
#     print(" r_inv_sqrt = ", r_inv_sqrt)
#     r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
#     print(" r_mat_inv_sqrt = ", r_mat_inv_sqrt)
#     return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
#
#
# def normalize_features(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     print("row sum = ", rowsum)
#
#     r_inv = np.power(rowsum, -1).flatten()
#     print(" r_inv = ", r_inv)
#     r_inv[np.isinf(r_inv)] = 0.
#     print(" r_inv = ", r_inv)
#
#     r_mat_inv = sp.diags(r_inv)
#     print(" r_mat_inv = ",r_mat_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#
# # test_sp = np.array([[0, 1, 0], [1, 0, 1]])
# # print(sp.csr_matrix(test_sp))
#
# content = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
# features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
# print(features.shape)
#
# my_features = sp.csr_matrix(test_sp,dtype=np.float32)
# print(my_features.shape)
# print(my_features)
#
# # print("dfasf dff = ", features)
# features = normalize_features(my_features)
# print("normalize_features = " ,features)

# adj = normalize_adj(adj + sp.eye(adj.shape[0]))

h = torch.tensor(np.array([[1.0, 2.0, 3.0],
                           [11.0, 22.0, 33.0]]))

h = h.to(torch.float32)
print("===== h ======= ")
a = nn.Parameter(torch.empty(size=(2 * 3, 1)))
nn.init.xavier_uniform_(a.data, gain=1.414)
print("===== a ======= ")

print(a)

wh = torch.matmul(h, a[:3, :])
wh2 = torch.matmul(h, a[3:, :])

print(wh)
print(wh2)

print(wh + wh2.T)

print(" =====  ")

input_concat = torch.cat([h.repeat(1, 2).view(2 * 2, -1), h.repeat(2, 1)], dim=1). \
    view(2, -1, 2 * 3)

print(torch.matmul(input_concat, a).squeeze(2))

"""
adj
"""


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


edges = np.array([[0, 1], [1, 2]])

adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(3, 3), dtype=np.float32)
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = adj + sp.eye(adj.shape[0])

print(" ====== adj ==========")
print(adj)
print(adj.toarray())
mx = adj
rowsum = np.array(mx.sum(1))
print(" ======== rowsum ===========")
print(rowsum)

r_inv_sqrt = np.power(rowsum, -0.5).flatten()
print(" =========== np.power(rowsum, -0.5) ========== ")
print(np.power(rowsum, -0.5))
print(r_inv_sqrt)

r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
print(" =========== r_inv_sqrt[np.isinf(r_inv_sqrt)] ============== ")
print(r_inv_sqrt)

r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
print(" ========== sp.diags(r_inv_sqrt) ============= ")
print(r_mat_inv_sqrt)
print(r_mat_inv_sqrt.toarray())

result = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

print(" ============== mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) ============= ")
print(result)
print(result.toarray())
