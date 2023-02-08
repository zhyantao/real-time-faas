import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


def show(job, job_name=None):
    """
    根据给定 job 绘制 DAG 图.

    :param job_name: 用户可指定 job 的名称
    :param job: 一个包含（或不包含）依赖关系的 job（DataFrame 格式）
    :return: 无返回值，绘制 DAG 图
    """
    if job_name is None:
        job_name = job['job_name'].loc[job.index[0]]

    task_name_list = job.loc[:, 'task_name']

    # 填充有向图 DiGraph 的属性
    G = nx.DiGraph()
    for task_name in task_name_list:
        node_name_list = task_name.split('_')
        node_name_len = len(node_name_list[0])
        node_name_list[0] = node_name_list[0][1:node_name_len]

        if not node_name_list[0].isdigit():
            G.add_node(task_name)
            continue

        curr_node_num = node_name_list[0]
        if len(node_name_list) == 1:
            G.add_node(curr_node_num)
        else:
            i = 1
            while i < len(node_name_list):
                dep_node_num = node_name_list[i]
                G.add_edge(dep_node_num, curr_node_num, weight=1.0)  # 边的权重统一设置为 1, 后面可以改
                i += 1

    # 展示图像
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title(job_name)
    nx.draw(G, with_labels=True, pos=nx.nx_agraph.graphviz_layout(G, prog='dot'), ax=ax)
    plt.show()


def merge(job1, job2):
    """
    合并两个 job1 和 job2，返回合并后的 job。

    :param job1:
    :param job2:
    :return: job（DataFrame 格式）
    """
    return pd.concat([job1, job2], axis=0)
