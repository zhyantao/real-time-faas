import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def show(job):
    df = pd.read_csv("E:/Workshop/real-time-faas/dataset/selected_batch_task.csv")
    task_name_list = df.loc[:, 'task_name']
    matrix_len = len(task_name_list)
    adj_matrix = np.zeros((matrix_len, matrix_len))  # 邻接矩阵
    for task_name in task_name_list:
        instance_name_list = task_name.split('_')
        instance_name_len = len(instance_name_list[0])
        instance_name = instance_name_list[0][1:instance_name_len]
        for i in range(1, len(instance_name_list)):
            adj_matrix[int(instance_name_list[i]), int(instance_name)] = 1.0  # 边的权重统一设置为 1, 后面可以改
    print(adj_matrix)
    dag = nx.DiGraph(adj_matrix)
    dag.remove_edges_from([edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0'])
    nx.draw(dag, pos=nx.nx_agraph.graphviz_layout(dag, prog='dot'), with_labels=True)
    plt.show()


def merge(job1, job2):
    pass


if __name__ == '__main__':
    show(job=None)
