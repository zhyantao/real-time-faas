import numpy as np


def load_job():
    adj_mat = np.load('../../../dataset/tpch/2g/adj_mat_1.npy', allow_pickle=True)
    task_durations = np.load('../../../dataset/tpch/2g/task_duration_1.npy', allow_pickle=True)
    print(adj_mat)
    print(task_durations)

    # (1) 根据 adj_mat 重建 DAG
    dags = rebuild_dag(adj_mat)

    # (2) 根据 task_durations 挑选 node
    durations = get_job_duration()

    # 后面应该抽象成一个类，每个 job 内都集成了 duration 和其他资源需求
    return dags, durations


def rebuild_dag(adj_mat):
    pass


def get_job_duration():
    pass


if __name__ == '__main__':
    load_job()
