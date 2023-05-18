import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from models.utils.parameters import args
from models.utils.tools import generate_random_numbers


class DAG:
    def __init__(self):
        pass

    @staticmethod
    def generate_dag_from_alibaba_trace_data(job: DataFrame, job_name=None):
        """
        根据给定 job 初始化 DAG.

        :param job_name: 用户可指定 job_name，不指定则由程序直接生成
        :param job: 一个包含（或不包含）依赖关系的 job（DataFrame 格式）
        """
        task_name_list = job.loc[:, 'task_name']
        # 填充有向图 DiGraph 的属性
        G = nx.DiGraph()
        for task_name in task_name_list:
            dependencies = task_name.split('_')
            task_name_len = len(dependencies[0])
            dependencies[0] = dependencies[0][1:task_name_len]

            if not dependencies[0].isdigit():
                G.add_node(task_name)
                continue

            curr_task = dependencies[0]
            if len(dependencies) == 1:
                G.add_node(curr_task)
            else:
                i, weights = 1, generate_random_numbers(len(dependencies) - 1)
                while i < len(dependencies):
                    dependency = dependencies[i]
                    G.add_edge(dependency, curr_task, weight=weights[i - 1])
                    i += 1

        # 重新为每条边分配随机权重（出度概率总和为 1）
        adj = nx.to_numpy_array(G)
        counts = np.count_nonzero(adj, axis=1)
        for i in range(len(adj)):
            weights, k = generate_random_numbers(counts[i]), 0
            for j in range(len(adj)):
                if adj[i, j] > 0:
                    adj[i, j] = weights[k]
                    k += 1
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

        if job_name is None:
            job_name = job['job_name'].loc[job.index[0]]

        # 保存任务间的依赖信息和转移概率信息
        np.save(args.task_depend_prefix + job_name + '_adj_matrix', adj)
        # 保存每个任务的资源需求信息：CPU、内存、需要处理的数据大小
        n_tasks = adj.shape[0]
        required_cpu = np.random.randint(args.required_cpu_lower, args.required_cpu_upper, n_tasks)
        np.save(args.task_depend_prefix + job_name + '_required_cpu', required_cpu)
        required_mem = np.random.randint(args.required_mem_lower, args.required_mem_upper, n_tasks)
        np.save(args.task_depend_prefix + job_name + '_required_mem', required_mem)
        data_size = np.random.randint(args.data_size_lower, args.data_size_upper, n_tasks)
        np.save(args.task_depend_prefix + job_name + '_data_size', data_size)
        duration = np.random.randint(args.duration_lower, args.duration_upper, n_tasks)
        np.save(args.task_depend_prefix + job_name + '_duration', duration)

        return G, job_name

    @staticmethod
    def merge(job1, job2):
        """
        合并两个 job1 和 job2，返回合并后的 job。

        :param job1: 一个包含（或不包含）依赖关系的 job（DataFrame 格式）
        :param job2: 另一个包含（或不包含）依赖关系的 job（DataFrame 格式）
        :return: job（DataFrame 格式）
        """
        return pd.concat([job1, job2], axis=0)
