"""
生成实验环境，主要包括任务间依赖信息和节点间的连通性信息。
"""
import numpy as np
import pandas as pd

from models.utils.dag import DAG
from models.utils.dataset import get_one_job
from models.utils.figure import DAGFigure, UDGFigure
from models.utils.params import args
from models.utils.udg import UDG


def gen_task_depends():
    """
    根据 alibaba 数据集重新生成任务间的依赖关系，同时包含了任务间的转移概率。
    """
    df = pd.read_csv(args.selected_batch_task_path)
    idx = 0
    while idx < df.shape[0]:
        job, idx = get_one_job(df, idx)
        G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
        dag_figure = DAGFigure()
        dag_figure.visual(G, job_name)


def gen_node_connects():
    """
    指定节点个数、最大连接数、带宽下限、带宽上限，以随机的方式生成节点间连通性的拓扑结构。
    """
    udg_figure = UDGFigure()
    udg = UDG()
    G, udg_name = udg.generate_udg_from_random(10, 5, 20, 60)
    udg_figure.visual(G, udg_name)


if __name__ == '__main__':
    gen_task_depends()
    gen_node_connects()
    b = np.load(args.node_connect_prefix + '10 nodes 5 connections.npy')
    print(b)
