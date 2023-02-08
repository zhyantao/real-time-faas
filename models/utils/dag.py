import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def show(job):
    """
    根据给定 job 绘制 DAG 图.

    :param job: 一个包含（或不包含）依赖关系的 job（DataFrame 格式）
    :return: 无返回值，绘制 DAG 图
    """
    job_name = job['job_name'].loc[job.index[0]]
    task_name_list = job.loc[:, 'task_name']
    node_nums = len(task_name_list)  # 该 job 包含的节点数量

    # 填充邻接矩阵
    adj_matrix = np.zeros((node_nums, node_nums))  # 邻接矩阵
    for task_name in task_name_list:
        node_name_list = task_name.split('_')
        node_name_len = len(node_name_list[0])
        node_name_list[0] = node_name_list[0][1:node_name_len]

        if not node_name_list[0].isdigit():
            continue

        curr_node_num = int(node_name_list[0])
        i = 1
        while i < len(node_name_list):
            dep_node_num = int(node_name_list[i])
            adj_matrix[dep_node_num - 1, curr_node_num - 1] = 1.0  # 边的权重统一设置为 1, 后面可以改
            i += 1

    # 使用邻接矩阵画图
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title(job_name)
    dag = nx.DiGraph(adj_matrix)
    dag.remove_edges_from([edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0'])
    nx.draw(dag, with_labels=True, pos=nx.nx_agraph.graphviz_layout(dag, prog='dot'), ax=ax)
    plt.show()


def merge(job1, job2):
    """
    合并两个 job1 和 job2，返回合并后的 job。

    :param job1:
    :param job2:
    :return: job（DataFrame 格式）
    """
    return pd.concat([job1, job2], axis=0)
