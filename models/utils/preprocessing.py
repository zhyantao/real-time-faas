"""
This script processes the Alibaba cluster trace dataset (v2018). v2018 has 6 CSV files. We only use the file
batch_task.csv.

- Firstly, we sample 2119 DAGs from the dataset and save them into selected_DAGs.csv.
- Then, we sort the functions of each DAG in topological order (used for DPE and FixDoc) and save the results to
    topological_order.csv.
- At last, we 'rank' the functions of each DAG (used for HEFT) and save the results to rank.scv.

(task --> function, func_num --> DAG)
"""
import os

import numpy as np
import pandas as pd

from models.utils.parameters import *
from models.utils.progress_bar import ProgressBar

bar = ProgressBar()


def sample_jobs(batch_task_path=BATCH_TASK_PATH, selected_batch_task_path=SELECTED_BATCH_TASK_PATH):
    """
    参考全局变量 REQUIRED_NUM:
    采样 200 个含有 2 个 task 的 sampled_jobs,
    800 个含有 3 ~ 10 个 task 的 sampled_jobs,
    600 个含有 11 ~ 50 个 task 的 sampled_jobs,
    400 个含有 51 ~ 100 个 task 的 sampled_jobs,
    119 个含有超过 100 个 task 的 sampled_jobs.

    :return: 将采样的 sampled_jobs 保存的文件中
    """
    if os.path.exists(selected_batch_task_path):
        print("batch_task & batch_instance is already selected.")
        return

    columns = ['task_name', 'instance_num', 'job_name', 'task_type', 'status',
               'start_time', 'end_time', 'plan_cpu', 'plan_mem']
    df_batch_task = pd.read_csv(batch_task_path, header=None, names=columns)

    required_num = REQUIRED_NUM
    counters = np.zeros(5)

    sampled_jobs = df_batch_task.loc[0: 0]  # 变量 sampled_jobs 用于保存采样出来的 sampled_jobs
    print('sampling jobs from batch task & batch instance ...')

    df_len_batch_task = df_batch_task.shape[0]
    idx = 0
    while idx < df_len_batch_task:
        # 跳过不含依赖关系的 jobs
        task_name = df_batch_task.loc[idx, 'task_name']
        if task_name.find('task_') != -1:
            idx = idx + 1
            continue

        # 将含有两个及以上 task 的 jobs 挑选出来
        job_name = df_batch_task.loc[idx, 'job_name']  # 每个 job 都是一个 DAG

        if not exist_in_batch_instance(job_name):
            continue

        task_nums = 0  # 该 job 包含的 task 数量
        while (idx + task_nums < df_len_batch_task) and (df_batch_task.loc[idx + task_nums, 'job_name'] == job_name):
            task_nums = task_nums + 1
        if task_nums == 2:
            if counters[0] < required_num[0]:
                sampled_jobs = pd.concat([sampled_jobs, df_batch_task.loc[idx: idx + task_nums - 1].copy()], axis=0)
                counters[0] = counters[0] + 1
        elif 3 <= task_nums <= 10:
            if counters[1] < required_num[1]:
                sampled_jobs = pd.concat([sampled_jobs, df_batch_task.loc[idx: idx + task_nums - 1].copy()], axis=0)
                counters[1] = counters[1] + 1
        elif 11 <= task_nums <= 50:
            if counters[2] < required_num[2]:
                sampled_jobs = pd.concat([sampled_jobs, df_batch_task.loc[idx: idx + task_nums - 1].copy()], axis=0)
                counters[2] = counters[2] + 1
        elif 51 <= task_nums <= 100:
            if counters[3] < required_num[3]:
                sampled_jobs = pd.concat([sampled_jobs, df_batch_task.loc[idx: idx + task_nums - 1].copy()], axis=0)
                counters[3] = counters[3] + 1
        elif task_nums > 100:
            if counters[4] < required_num[4]:
                sampled_jobs = pd.concat([sampled_jobs, df_batch_task.loc[idx: idx + task_nums - 1].copy()], axis=0)
                counters[4] = counters[4] + 1
        idx = idx + task_nums

        # 更新进度条
        percent = sum(counters) / float(sum(required_num)) * 100
        bar.update(percent)

        # 如果已经满足了采样数量，提前终止程序
        if sum(counters) == all:
            break

    sampled_jobs.to_csv(selected_batch_task_path, index=False)


def exist_in_batch_instance(job_name, batch_instance_path=BATCH_INSTANCE_PATH,
                            selected_batch_instance_path=SELECTED_BATCH_INSTANCE_PATH):
    """
    找出 batch_instance.csv 中与 batch_task.csv 中 task_name 一致的 jobs

    :param job_name:
    :param batch_instance_path:
    :param selected_batch_instance_path:
    :return: 将查找结果输出到文件中
    """

    columns = ['instance_name', 'task_name', 'job_name', 'task_type', 'status',
               'start_time', 'end_time', 'machine_id', 'seq_no', 'total_seq_no',
               'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
    df_batch_instance = pd.read_csv(batch_instance_path, header=None, names=columns)

    sampled_jobs = df_batch_instance.loc[0: 0]  # 变量 sampled_jobs 用于保存采样出来的 sampled_jobs

    df_len_batch_instance = df_batch_instance.shape[0]
    idx = 0

    # 在 batch_instance.csv 中查找名为 job_name 的 job, 将其挑选出来
    while idx < df_len_batch_instance:
        instance_nums = 0
        while (idx + instance_nums) < df_len_batch_instance \
                and job_name == df_batch_instance.loc[idx + instance_nums, 'job_name']:
            instance_nums = instance_nums + 1
        if instance_nums == 0:
            idx = idx + 1
        else:
            sampled_jobs = pd.concat([sampled_jobs, df_batch_instance.loc[idx: idx + instance_nums - 1].copy()], axis=0)
            break

    if not sampled_jobs.empty:
        sampled_jobs.to_csv(selected_batch_instance_path, index=False)
        return True  # 如果 batch_instance 中有和 batch_task 一致的 task_name, 返回 True
    else:
        return False


def get_topological_order(selected_DAG_path=SELECTED_DAG_PATH, sorted_DAG_path=SORTED_DAG_PATH):
    """
    Get the topological ordering of each DAG, save the results into the file topological_order.csv.
    """
    if os.path.exists(sorted_DAG_path):
        print('DAGs\' topological order has been obtained!')
        return

    if not os.path.exists(selected_DAG_path):
        print('The sampling procedure has not been executed! Please sampling DAGs firstly.')
        return

    df = pd.read_csv(selected_DAG_path)
    df_len = df.shape[0]  # CSV 文件的行数
    idx = 0

    required_num = REQUIRED_NUM
    all_DAG_num = sum(required_num)
    sorted_num = 0

    print('Getting topological order for %d DAGs...' % all_DAG_num)
    while idx < df_len:  # 遍历 CSV 文件的每一行
        # 获取一个 DAG
        job_name = df.loc[idx, 'job_name']
        DAG_len = 0  # DAG 中包含的 task 数目
        while (idx + DAG_len < df_len) and (df.loc[idx + DAG_len, 'job_name'] == job_name):
            DAG_len = DAG_len + 1
        DAG = df.loc[idx: idx + DAG_len].copy()

        # get the number and dependencies of each function of the DAG
        funcs_num = np.zeros(DAG_len)  # 函数数量
        dependencies = [[] * 1] * DAG_len
        for i in range(DAG_len):
            name_str_list = DAG.loc[i + idx, 'task_name'].split('_')
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
        funcs_left = DAG_len  # 剩余未排序的函数数目
        DAG_sorted = DAG.copy()
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
            for i in range(len(dependencies)):
                if len(dependencies[i]) == 0:
                    running_func = i
                    dependencies[i].append(-1)
                    break
            func_running = int(funcs_num[running_func])
            for i in range(len(dependencies)):
                if dependencies[i].count(func_running) > 0:
                    dependencies[i].remove(func_running)
            DAG_sorted.loc[DAG_len - funcs_left + idx] = DAG.loc[running_func + idx].copy()
            funcs_left = funcs_left - 1

        df.loc[idx: idx + DAG_len - 1] = DAG_sorted.copy()  # 保存排序结果

        idx = idx + DAG_len  # 遍历下一个 DAG

        # 使用进度条显示当前的处理进度
        sorted_num = sorted_num + 1
        percent = sorted_num / float(all_DAG_num) * 100
        # for overflow
        if percent > 100:
            percent = 100
        bar.update(percent)

    df.to_csv(sorted_DAG_path, index=False)
