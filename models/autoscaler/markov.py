from collections import deque

import networkx as nx
import numpy as np
import pandas as pd

from models.autoscaler.dag import DAG
from models.autoscaler.figure import DAGFigure
from models.utils.dataset import get_one_job
from models.utils.params import args


class MostLikelyPath:
    def __init__(self):
        self.path = deque()
        self.visited = set()

    def markov_predict_path(self, transition_matrix, step):
        """
        :param step: 状态转移的步数
        :param transition_matrix: 状态转移矩阵
        :return:
        """
        # 访问 NumPy 二维数组中的元素用 [i, j] 而不是 [i][j]

        # 初始状态向量（从入度为 0 的开始转移）
        all_zero_cols = np.all(transition_matrix == 0, axis=0)
        initial_state = all_zero_cols.astype(int)
        # print('initial_state: ', initial_state)

        # 预测未来的状态
        for i in range(step):
            predicted_state = initial_state.dot(transition_matrix)
            # print('predicted_state ', predicted_state)

            # 使用 argsort() 函数对数组的索引进行排序
            sorted_indices = np.argsort(predicted_state, axis=1)
            # print('sorted_indices ', sorted_indices)

            n = sorted_indices.shape[1]
            for j in range(n - 1, -1, -1):
                curr_index = sorted_indices[0, j]
                if predicted_state[0, curr_index] == 0:
                    break
                self.put(curr_index)  # 找到最可能的状态

            initial_state = predicted_state  # 更新为下一个状态

        return np.array(self.path)

    def put(self, item):
        if item not in self.visited:
            self.path.append(item)
            self.visited.add(item)

    def get(self):
        return self.path.popleft()

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    # df = pd.read_csv(args.selected_batch_task_path)
    #
    # # 遍历一个 job
    # job, idx = get_one_job(df)
    # G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
    # DAGFigure().visual(G)  # 可视化 job
    # adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    # mlp = MostLikelyPath()
    # path = mlp.markov_predict_path(adj, 3)
    # print(job_name, '\t', path)
    #
    # # 遍历另一个 job
    # job, idx = get_one_job(df, idx)
    # G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
    # DAGFigure().visual(G)  # 可视化 job
    # adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    # mlp = MostLikelyPath()
    # path = mlp.markov_predict_path(adj, 3)
    # print(job_name, ' ', path)

    df = pd.read_csv(args.selected_batch_task_path)
    idx = 0
    while idx < df.shape[0]:
        job, idx = get_one_job(df, idx)
        G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
        dag_figure = DAGFigure()
        dag_figure.visual(G, job_name)
        adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
        mlp = MostLikelyPath()
        path = mlp.markov_predict_path(adj, 3)
        print(job_name, '\t', path)
