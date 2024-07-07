"""
从 batch_task.csv 和 batch_instance 中 1) 提取 100 个 job, 并进行 2) 拓扑排序, 将 1 和 2 的结果保存到文件中.

- job: 对应本文的（用户请求），不含依赖
- task: 对应本文的 function，含依赖
- instance: 对应本文的容器（Docker），不含依赖

注：一个 job 含有若干 task，一个 task 含有若干 instance.
"""
import os
import re

import numpy as np
import pandas as pd

from models.utils.dag import DAG
from models.utils.figure import DAGFigure, UDGFigure
from models.utils.parameters import args
from models.utils.text import ProgressBar
from models.utils.tools import get_one_job
from models.utils.udg import UDG


def download_batch_task():
    """
    从网络上下载 batch_task.tar.gz，并保存到 dataset 目录下
    """
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    if not os.path.exists('dataset/batch_task.tar.gz'):
        os.system('wget http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz -P dataset')
        os.system('tar -zxvf dataset/batch_task.tar.gz --directory=dataset')

    if not os.path.exists('dataset/batch_instance.tar.gz'):
        os.system('wget http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_instance.tar.gz -P dataset')
        os.system('tar -zxvf dataset/batch_instance.tar.gz --directory=dataset')

    if not os.path.exists('dataset/container_usage.tar.gz'):
        os.system('wget http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz -P dataset')
        os.system('tar -zxvf dataset/container_usage.tar.gz --directory=dataset')

def sample_jobs(batch_task_path=args.batch_task_path,
                selected_batch_task_path=args.selected_batch_task_path,
                batch_instance_path=args.batch_instance_path,
                selected_batch_instance_path=args.selected_batch_instance_path):
    """
    :return: 从 batch_task.csv 和 batch_instance.csv 中提取 100 个 job_name 相同的 jobs, 保存到文件中
    """
    if os.path.exists(selected_batch_task_path):
        print("Dataset (batch_task.csv & batch_instance.csv) is already selected.")
        return

    if not os.path.exists(selected_batch_instance_path):
        columns = ['instance_name', 'task_name', 'job_name', 'task_type', 'status',
                   'start_time', 'end_time', 'machine_id', 'seq_no', 'total_seq_no',
                   'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        df = pd.DataFrame(columns=columns)
        df.to_csv(selected_batch_instance_path, index=False)

    if not os.path.exists(selected_batch_task_path):
        columns = ['task_name', 'instance_num', 'job_name', 'task_type', 'status',
                   'start_time', 'end_time', 'plan_cpu', 'plan_mem']
        df = pd.DataFrame(columns=columns)
        df.to_csv(selected_batch_task_path, index=False)

    print("Sampling data from batch_task.csv & batch_instance.csv ...")

    chunk_size = 3
    idx_batch_instance, idx_batch_task = 0, 0

    chunk_batch_task = pd.read_csv(batch_task_path, header=None, iterator=True)
    chunk_batch_instance = pd.read_csv(batch_instance_path, header=None, iterator=True)

    current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
    current_chunk_batch_instance = chunk_batch_instance.get_chunk(chunk_size)

    i, j, count = 0, 0, 0  # count 是当前已经采集的 job 数量, total 是总共的
    bar = ProgressBar()
    while (not current_chunk_batch_instance.empty) and (not current_chunk_batch_task.empty):

        job_name_batch_task = current_chunk_batch_task.loc[idx_batch_task + i, 2]
        job_name_num_batch_task = int(re.findall(r"\d+\.?\d*", job_name_batch_task)[0])

        job_name_batch_instance = current_chunk_batch_instance.loc[idx_batch_instance + j, 2]
        job_name_num_batch_instance = int(re.findall(r"\d+\.?\d*", job_name_batch_instance)[0])

        if job_name_num_batch_task == job_name_num_batch_instance:
            job_name_num_batch_task_tmp = job_name_num_batch_task
            # First 保存 batch task
            while i < chunk_size:
                item_batch_task = current_chunk_batch_task.loc[idx_batch_task + i: idx_batch_task + i]
                # print(item_batch_task)
                with open(selected_batch_task_path, 'a') as f:
                    item_batch_task.to_csv(f, header=False, index=False, lineterminator="\n")

                i += 1
                if i == chunk_size:
                    current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
                    idx_batch_task += chunk_size
                    i = 0

                job_name_batch_task = current_chunk_batch_task.loc[idx_batch_task + i, 2]
                job_name_num_batch_task = int(re.findall(r"\d+\.?\d*", job_name_batch_task)[0])

                if job_name_num_batch_task != job_name_num_batch_instance:
                    break

            # Second 保存 batch instance
            while j < chunk_size:
                item_batch_instance = current_chunk_batch_instance.loc[idx_batch_instance + j: idx_batch_instance + j]
                # print(item_batch_instance)
                with open(selected_batch_instance_path, 'a') as f:
                    item_batch_instance.to_csv(f, header=False, index=False, lineterminator="\n")

                j += 1
                if j == chunk_size:
                    current_chunk_batch_instance = chunk_batch_instance.get_chunk(chunk_size)
                    idx_batch_instance += chunk_size
                    j = 0

                job_name_batch_instance = current_chunk_batch_instance.loc[idx_batch_instance + j, 2]
                job_name_num_batch_instance = int(re.findall(r"\d+\.?\d*", job_name_batch_instance)[0])

                if job_name_num_batch_task_tmp != job_name_num_batch_instance:
                    break

            count += 1
            percent = count / float(args.total_jobs) * 100
            bar.update(percent)
            if count == args.total_jobs:
                return

        elif job_name_num_batch_task < job_name_num_batch_instance:
            while i < chunk_size:
                i += 1
                if i == chunk_size:
                    current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
                    idx_batch_task += chunk_size
                    i = 0

                job_name_batch_task = current_chunk_batch_task.loc[idx_batch_task + i, 2]
                job_name_num_batch_task = int(re.findall(r"\d+\.?\d*", job_name_batch_task)[0])

                if job_name_num_batch_task >= job_name_num_batch_instance:
                    break

        elif job_name_num_batch_task > job_name_num_batch_instance:
            while j < chunk_size:
                j += 1
                if j == chunk_size:
                    current_chunk_batch_instance = chunk_batch_instance.get_chunk(chunk_size)
                    idx_batch_instance += chunk_size
                    j = 0

                job_name_batch_instance = current_chunk_batch_instance.loc[idx_batch_instance + j, 2]
                job_name_num_batch_instance = int(re.findall(r"\d+\.?\d*", job_name_batch_instance)[0])

                if job_name_num_batch_task <= job_name_num_batch_instance:
                    break


def get_topological_order(selected_batch_task_path=args.selected_batch_task_path,
                          batch_task_topological_order_path=args.batch_task_topological_order_path):
    """
    获取每个 job 中 task 的拓扑排序，保存到文件中。
    """
    if os.path.exists(batch_task_topological_order_path):
        print('Jobs\' topological order (batch_task_topological_order.csv) has been obtained.')
        return

    if not os.path.exists(selected_batch_task_path):
        print('The sampling procedure has not been executed! Please sampling jobs firstly.')
        return

    df = pd.read_csv(selected_batch_task_path)
    rows = df.shape[0]  # CSV 文件的行数
    idx = 0

    total_jobs = args.total_jobs
    sorted_num = 0
    bar = ProgressBar()

    print('Getting topological order for %d jobs ...' % total_jobs)
    while idx < rows:  # 遍历 CSV 文件的每一行

        task_name = df.loc[idx, 'task_name'].split('_')[0]

        if task_name == 'task' or task_name == 'MergeTask':
            job_name = df.loc[idx, 'job_name']
            task_nums = 0  # job 中包含的 task 数目
            while (idx + task_nums < rows) and (df.loc[idx + task_nums, 'job_name'] == job_name):
                task_nums = task_nums + 1
            idx = idx + task_nums

        else:
            # 获取一个 job
            job_name = df.loc[idx, 'job_name']
            task_nums = 0  # job 中包含的 task 数目
            while (idx + task_nums < rows) and (df.loc[idx + task_nums, 'job_name'] == job_name):
                task_nums = task_nums + 1
            job = df.loc[idx: idx + task_nums].copy()

            # get the number and dependencies of each function of the job
            funcs_num = np.zeros(task_nums)  # 函数数量
            dependencies = [[] * 1] * task_nums
            for i in range(task_nums):
                name_str_list = job.loc[i + idx, 'task_name'].split('_')
                name_str_list_len = len(name_str_list)
                func_str_len = len(name_str_list[0])
                func_num = int(name_str_list[0][1:func_str_len])  # 函数编号
                dependent_funcs = []
                for j in range(name_str_list_len):
                    if j == 0:  # 跳过函数自身
                        # the func itself
                        continue
                    if name_str_list[j].isnumeric():
                        # the function's dependencies
                        dependent_func_num = int(name_str_list[j])
                        dependent_funcs.append(dependent_func_num)
                funcs_num[i] = func_num
                dependencies[i] = dependent_funcs

            # sort the functions according to their dependencies
            funcs_left = task_nums  # 剩余未排序的函数数目
            job_sorted = job.copy()
            while funcs_left > 0:
                # find a source func, and place the funcs who depend on it after this source func
                # the topological ordering we take is actually a Depth-first Search algorithm
                # as a result, the entry functions may not have the smallest number
                #
                # 以一个元函数为基准，任何依赖该函数的其他函数置于其后
                # 我们使用了 DFS 算法进行拓扑排序，导致入口函数可能不是最小的数字

                # ==== this is where we can improved ====
                # Use Breadth-first Search algorithm to obtain the topological ordering and compare the results.
                # The makespan might be decreased further.
                #
                # 使用 BFS 算法进行拓扑排序可以进一步降低 makespan
                # =======================================
                running_func = -1
                for i in range(len(dependencies)):
                    if len(dependencies[i]) == 0:
                        running_func = i
                        dependencies[i].append(-1)
                        break
                func_running = int(funcs_num[running_func])
                for i in range(len(dependencies)):
                    if dependencies[i].count(func_running) > 0:
                        dependencies[i].remove(func_running)
                job_sorted.loc[task_nums - funcs_left + idx] = job.loc[running_func + idx].copy()
                funcs_left = funcs_left - 1

            df.loc[idx: idx + task_nums - 1] = job_sorted.copy()  # 保存排序结果

            idx = idx + task_nums  # 遍历下一个 job

        # 使用进度条显示当前的处理进度
        sorted_num = sorted_num + 1
        percent = sorted_num / float(total_jobs) * 100
        # for overflow
        if percent > 100:
            percent = 100
        bar.update(percent)

    df.to_csv(batch_task_topological_order_path, index=False)


def sample_machines(container_usage_path=args.container_usage_path,
                    selected_container_usage_path=args.selected_container_usage_path):
    """
    从 container_usage.csv 文件中提取指定数量的 machines

    :return: 无返回值（保存到文件）
    """
    if os.path.exists(selected_container_usage_path):
        print("Dataset container_usage.csv is already selected.")
        return

    if not os.path.exists(selected_container_usage_path):
        columns = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent',
                   'cpi', 'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']
        df = pd.DataFrame(columns=columns)
        df.to_csv(selected_container_usage_path, index=False)

    print("Sampling data from container_usage.csv ...")

    chunk_size = 3
    chunk_container_usage = pd.read_csv(container_usage_path, header=None, iterator=True)
    current_chunk_container_usage = chunk_container_usage.get_chunk(chunk_size)

    idx, i, count = 0, 0, 0
    bar = ProgressBar()
    while not current_chunk_container_usage.empty:
        machine_id = current_chunk_container_usage.loc[idx + i, 1]

        while i < chunk_size:
            if machine_id == current_chunk_container_usage.loc[idx + i, 1]:
                item_container_usage = current_chunk_container_usage.loc[idx + i: idx + i]
                with open(selected_container_usage_path, 'a') as f:
                    item_container_usage.to_csv(f, header=False, index=False, lineterminator='\n')
                i += 1

                if i == chunk_size:
                    current_chunk_container_usage = chunk_container_usage.get_chunk(chunk_size)
                    idx += chunk_size
                    i = 0
            else:
                machine_id = current_chunk_container_usage.loc[idx + i, 1]  # 更新为下一个 machine_id

                count += 1
                percent = count / float(args.total_machines) * 100
                bar.update(percent)
                if count == args.total_machines:
                    return


def gen_task_depend_input():
    """
    根据 alibaba 数据集重新生成任务间的依赖关系，同时包含了任务间的转移概率
    :return:
    """
    df = pd.read_csv(args.selected_batch_task_path)
    idx = 0
    while idx < df.shape[0]:
        job, idx = get_one_job(df, idx)
        G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
        dag_figure = DAGFigure()
        dag_figure.visual(G, job_name)


def gen_heft_and_dpe_input():
    """
    根据节点间的连通性信息，生成 HEFT 算法所需要的环境输入。
    :return:
    """
    # 获取节点间的拓扑结构信息
    udg_name = str(args.n_nodes) + '_nodes_' + str(args.n_max_connections) + '_connections'
    npy_file_name_prefix = args.node_connect_prefix + udg_name
    adj_matrix = np.load(npy_file_name_prefix + '_adj_matrix.npy')
    G = np.where(adj_matrix != 0, 1, 0)  # 把非零元素替换为 1
    bw = adj_matrix
    pp = np.load(npy_file_name_prefix + '_processing_power.npy')
    return G, bw, pp


def gen_node_connect_input():
    """
    指定节点个数、最大连接数、带宽下限、带宽上限，以随机的方式生成节点间连通性的拓扑结构。
    另外还需包含每个计算节点处理数据的能力大小。
    :return:
    """
    n_nodes = args.n_nodes
    n_max_connections = args.n_max_connections
    bandwidth_lower = args.bandwidth_lower
    bandwidth_upper = args.bandwidth_upper
    processing_power_lower = args.processing_power_lower
    processing_power_upper = args.processing_power_upper

    # 随机生成节点之间的连通性信息以及节点间的带宽大小，并进行可视化
    udg_figure = UDGFigure()
    udg = UDG()
    G, udg_name = udg.generate_udg_from_random(n_nodes, n_max_connections, bandwidth_lower, bandwidth_upper)
    udg_figure.visual(G, udg_name)

    # 随机生成每个节点的处理能力
    nodes_processing_power = np.random.randint(processing_power_lower, processing_power_upper, n_nodes)
    npy_file_name = udg_name.replace(' ', '_') + '_processing_power'
    np.save(args.node_connect_prefix + npy_file_name, nodes_processing_power)


def gen_gcn_dqn_input():
    """
    根据 task_depend_* 和 node_connect_* 加载 GCN-DQN 算法的输入。
    需要准备的数据包括：（参考 https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L24）
    - 一个 N * N 的邻接矩阵（N 是 tasks 的数量）
    - 一个 N * D 的特征矩阵（D 是每个 task 的特征数量）
    - 一个 N * E 的独热编码矩阵（E 是 nodes 数量）
    :return:
    """
    job_name = 'j_82634'
    task_adj_matrix = np.load(args.task_depend_prefix + job_name + '_adj_matrix.npy')
    task_data_size = np.load(args.task_depend_prefix + job_name + '_data_size.npy')
    task_duration = np.load(args.task_depend_prefix + job_name + '_duration.npy')
    task_required_cpu = np.load(args.task_depend_prefix + job_name + '_required_cpu.npy')
    task_required_mem = np.load(args.task_depend_prefix + job_name + '_required_mem.npy')

    udg_name = str(args.n_nodes) + '_nodes_' + str(args.n_max_connections) + '_connections'
    npy_file_name_prefix = args.node_connect_prefix + udg_name
    node_adj_matrix = np.load(npy_file_name_prefix + '_adj_matrix.npy')
    node_processing_power = np.load(npy_file_name_prefix + '_processing_power.npy')

    task_features = np.array([task_data_size, task_duration, task_required_cpu, task_required_mem]).T
    node_features = np.array([node_processing_power]).T

    return task_adj_matrix, task_features, node_adj_matrix, node_features


if __name__ == '__main__':
    gen_gcn_dqn_input()
