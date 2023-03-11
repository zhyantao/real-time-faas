import networkx as nx
import pandas as pd

from models.autoscaler.dag import DAG
from models.autoscaler.figure import DAGFigure
from models.utils.dataset import get_one_job
from models.utils.params import args


def markov_predict_path(graph, start, end):
    """
    This code takes as input a DAG represented as an adjacency matrix graph,
    the start node start, and the end node end. It then uses the Markov Chain
    algorithm to traverse the graph and compute the probability of each path
    from start to end. Finally, it returns the most likely path from start to end.

    Note that the Markov Chain algorithm assumes that the probabilities of
    transitioning from one node to another are constant and independent of
    previous transitions. This assumption may not hold in all cases, and the
    algorithm may not always produce the correct result.

    :param graph:
    :param start:
    :param end:
    :return:
    """

    # Initialize variables
    n = len(graph)
    visited = [False] * n
    path_prob = [0] * n
    path_prob[start] = 1
    path = [[] for _ in range(n)]
    path[start] = [start]
    print(path)

    # Traverse the graph
    for i in range(n):
        max_prob = -1
        max_node = -1
        for j in range(n):
            if not visited[j] and path_prob[j] > max_prob:
                max_prob = path_prob[j]
                max_node = j
        if max_node == -1:
            break
        visited[max_node] = True
        for k in range(n):
            if graph[max_node][k] != 0:
                new_prob = path_prob[max_node] * graph[max_node][k]
                if new_prob > path_prob[k]:
                    path_prob[k] = new_prob
                    path[k] = path[max_node] + [k]

    # Return the most likely path from start to end
    return path[end]


if __name__ == '__main__':
    df = pd.read_csv(args.selected_batch_task_path)

    # 遍历一个 job
    job, idx = get_one_job(df)
    DAGFigure().visual(job)  # 可视化 job
    G, _ = DAG.generate_dag_from_alibaba_trace_data(job)
    adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    print(adj)
    # path = markov_predict_path(adj, 1, 1)
    # print(path)

    # 遍历另一个 job
    job, idx = get_one_job(df, idx)
    DAGFigure().visual(job)  # 可视化 job
    G, _ = DAG.generate_dag_from_alibaba_trace_data(job)
    adj = nx.to_numpy_matrix(G)  # 将 DiGraph 转为邻接矩阵
    print(adj)
