import networkx as nx
import pandas as pd

from models.autoscaler.dag import DAG
from models.autoscaler.figure import DAGFigure
from models.utils.dataset import get_one_job
from models.utils.params import args


def markov_predict_path(graph, start, end):
    """
    :param graph:
    :param start:
    :param end:
    :return:
    """
    # 访问 NumPy 二维数组中的元素用 [i, j] 而不是 [i][j]
    n = len(graph)
    return [i for i in range(start, end)]


def dfs(matrix, visited, node):
    visited[node] = True
    print("Visited node:", node)
    for i in range(len(matrix)):
        if matrix[node, i] > 0 and not visited[i]:
            dfs(matrix, visited, i)


if __name__ == '__main__':
    df = pd.read_csv(args.selected_batch_task_path)

    # 遍历一个 job
    job, idx = get_one_job(df)
    G, _ = DAG.generate_dag_from_alibaba_trace_data(job)
    DAGFigure().visual(G)  # 可视化 job
    adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    print(adj)
    path = markov_predict_path(adj, 0, 1)
    print(path)

    # 遍历另一个 job
    job, idx = get_one_job(df, idx)
    G, _ = DAG.generate_dag_from_alibaba_trace_data(job)
    DAGFigure().visual(G)  # 可视化 job
    adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    print(adj)
    path = markov_predict_path(adj, 1, 8)
    print(path)
