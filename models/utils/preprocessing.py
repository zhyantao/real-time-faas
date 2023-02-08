"""
从 batch_task.csv 和 batch_instance 中 1) 提取 100 个 job, 并进行 2) 拓扑排序, 将 1 和 2 的结果保存到文件中.

- job: 对应本文的（用户请求），不含依赖
- task: 对应本文的 function，含依赖
- instance: 对应本文的容器（Docker），不含依赖

注：一个 job 含有若干 task，一个 task 含有若干 instance.
"""
import re

import numpy as np
import pandas as pd

from models.utils.parameters import *
from models.utils.progress_bar import ProgressBar

bar = ProgressBar()


def sample_jobs(batch_task_path=BATCH_TASK_PATH, selected_batch_task_path=SELECTED_BATCH_TASK_PATH,
                batch_instance_path=BATCH_INSTANCE_PATH,
                selected_batch_instance_path=SELECTED_BATCH_INSTANCE_PATH):
    """
    :return: 从 batch_task.csv 和 batch_instance.csv 中提取 100 个 job_name 相同的 jobs, 保存到文件中
    """
    if os.path.exists(selected_batch_task_path):
        print("batch_task & batch_instance is already selected.")
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

    i, j, count, total = 0, 0, 0, 100  # count 是当前已经采集的 job 数量, total 是总共的
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
            percent = count / float(total) * 100
            bar.update(percent)
            if count == total:
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


if __name__ == "__main__":
    sample_jobs()
