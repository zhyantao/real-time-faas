"""计算节点之间的相似度"""

import numpy as np


def pagerank(adj_matrix, alpha=0.85, max_iter=100, tol=1e-6):
    """

    :param adj_matrix: 邻接矩阵
    :param alpha: 阻尼系数
    :param max_iter: 最大迭代次数
    :param tol: 收敛精度
    :return: 返回每个节点的 page_rank 值
    """
    assert adj_matrix.shape[0] == adj_matrix.shape[1]  # 确保是一个方阵
    n = adj_matrix.shape[0]
    d = np.sum(adj_matrix, axis=1)
    d[d == 0] = 1
    P = adj_matrix / d[:, np.newaxis]
    v = np.ones(n) / n
    for i in range(max_iter):
        new_v = alpha * np.dot(P, v) + (1 - alpha) / n
        if np.linalg.norm(new_v - v) < tol:
            break
        v = new_v
        print(v)
    # 处理孤立节点
    v[d == 0] = 1 / n
    return v


if __name__ == '__main__':
    ranks = pagerank(np.random.randint(10, size=(20, 20)))
    print(ranks)
