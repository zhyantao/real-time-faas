import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def show(job):
    job_name = job.loc[idx, 'job_name']
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
    pass


if __name__ == '__main__':
    df_batch_task = pd.read_csv("E:/Workshop/real-time-faas/dataset/selected_batch_task.csv")
    df_rows = df_batch_task.shape[0]
    idx = 0
    while idx < df_rows:
        job_name = df_batch_task.loc[idx, 'job_name']
        task_nums = 0
        while (idx + task_nums < df_rows) and (job_name == df_batch_task.loc[idx + task_nums, 'job_name']):
            task_nums += 1
        job = df_batch_task.loc[idx: idx + task_nums - 1].copy()
        show(job)  # 绘制 DAG 图
        idx += task_nums
