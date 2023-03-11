import networkx as nx
import pandas as pd
from pandas import DataFrame

from models.autoscaler.utils import generate_random_numbers


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
                i, weight = 1, generate_random_numbers(len(dependencies) - 1)
                while i < len(dependencies):
                    dependency = dependencies[i]
                    # G.add_edge(dependency, curr_task, weight=weight[i - 1])  # 为了与 batch.csv 中编号匹配可解开此注释
                    G.add_edge(int(dependency) - 1, int(curr_task) - 1, weight=weight[i - 1])  # 为了编程方便
                    i += 1

        if job_name is None:
            job_name = job['job_name'].loc[job.index[0]]

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


if __name__ == '__main__':
    print(generate_random_numbers(1))
