"""
This script processes the Alibaba cluster trace dataset (v2018). v2018 has 6 CSV files. We only use the file
batch_task.csv.

- Firstly, we sample 2119 DAGs from the dataset and save them into selected_DAGs.csv.
- Then, we sort the functions of each DAG in topological order (used for DPE and FixDoc) and save the results to
    topological_order.csv.
- At last, we 'rank' the functions of each DAG (used for HEFT) and save the results to rank.scv.

(task --> function, func_num --> DAG)
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
    参考全局变量 REQUIRED_NUM:
    采样 20 个含有 2 个 task 的 sampled_jobs_batch_task,
    80 个含有 3 ~ 10 个 task 的 sampled_jobs_batch_task,
    60 个含有 11 ~ 50 个 task 的 sampled_jobs_batch_task,
    40 个含有 51 ~ 100 个 task 的 sampled_jobs_batch_task,
    19 个含有超过 100 个 task 的 sampled_jobs_batch_task.

    :return: 将采样的 sampled_jobs_batch_task 保存的文件中
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

    required_num = REQUIRED_NUM
    counters = np.zeros(5)

    chunk_size = 3
    idx_batch_instance, idx_batch_task = 0, 0
    chunk_batch_instance = pd.read_csv(batch_instance_path, header=None, iterator=True)
    chunk_batch_task = pd.read_csv(batch_task_path, header=None, iterator=True)
    current_chunk_batch_instance = chunk_batch_instance.get_chunk(chunk_size)
    current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
    while not current_chunk_batch_instance.empty:
        i = 0
        while i < chunk_size:
            # 遍历 current_chunk_batch_instance
            job_name_batch_instance = current_chunk_batch_instance.loc[idx_batch_instance + i, 2]
            job_name_num_batch_instance = int(re.findall(r"\d+\.?\d*", job_name_batch_instance)[0])

            while not current_chunk_batch_task.empty:
                break_flag = False
                j = 0
                while j < chunk_size:
                    # 遍历 current_chunk_batch_task
                    job_name_batch_task = current_chunk_batch_task.loc[idx_batch_task + j, 2]
                    job_name_num_batch_task = int(re.findall(r"\d+\.?\d*", job_name_batch_task)[0])

                    print(job_name_num_batch_instance, job_name_num_batch_task)

                    # Case 1
                    # 如果 current_chunk_batch_task 中的 job_name == current_chunk_batch_instance 中的 job_name
                    # 循环记录，这里比较复杂
                    if job_name_num_batch_task == job_name_num_batch_instance:
                        print(current_chunk_batch_instance)
                        print(current_chunk_batch_task)

                        # 保存逻辑实现
                        sampled_jobs_batch_task = pd.DataFrame()  # 存储采样出来的 jobs
                        sampled_jobs_batch_instance = pd.DataFrame()

                        # 保存 batch_task.csv 中的 job
                        task_nums = 0
                        while j + task_nums < chunk_size and job_name_batch_task == \
                                current_chunk_batch_task.loc[idx_batch_task + j + task_nums, 2]:
                            task_nums += 1
                        sampled_jobs_batch_task = pd.concat(
                            [sampled_jobs_batch_task,
                             current_chunk_batch_task.loc[
                             idx_batch_task + j: idx_batch_task + j + task_nums - 1].copy()],
                            axis=0)
                        with open(selected_batch_task_path, 'a') as f:
                            sampled_jobs_batch_task.to_csv(f, header=False, index=False, lineterminator="\n")
                        j += task_nums

                        # 保存 batch_instance.csv 中的 job
                        instance_nums = 0
                        while i + instance_nums < chunk_size and job_name_batch_task == \
                                current_chunk_batch_instance.loc[idx_batch_instance + i + instance_nums, 2]:
                            instance_nums += 1
                        sampled_jobs_batch_instance = pd.concat(
                            [sampled_jobs_batch_instance,
                             current_chunk_batch_instance.loc[
                             idx_batch_instance + i: idx_batch_instance + i + instance_nums - 1].copy()],
                            axis=0)
                        with open(selected_batch_instance_path, 'a') as f:
                            sampled_jobs_batch_instance.to_csv(f, header=False, index=False, lineterminator="\n")
                        i += instance_nums

                        if i == chunk_size:
                            break_flag = True

                        if j == chunk_size:
                            current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
                            idx_batch_task += chunk_size

                        break

                    # Case 2
                    # 如果 current_chunk_batch_task 中的 job_name < current_chunk_batch_instance 中的 job_name
                    # 当前循环继续，continue，让 job_name_num 继续增大
                    if job_name_num_batch_task < job_name_num_batch_instance:
                        j += 1
                        continue

                    # Case 3
                    # 如果 current_chunk_batch_task 中的 job_name > current_chunk_batch_instance 中的 job_name
                    # 跳出内层循环，让外层循环的 job_name_num 继续增大
                    if job_name_num_batch_task > job_name_num_batch_instance:
                        break_flag = True
                        break

                if break_flag:
                    break

                # 保持内层循环的 job_name 持续增大
                current_chunk_batch_task = chunk_batch_task.get_chunk(chunk_size)
                idx_batch_task += chunk_size

            # Outer loop
            i += 1

        # 保持外层循环的 job_name 持续增大
        current_chunk_batch_instance = chunk_batch_instance.get_chunk(chunk_size)
        idx_batch_instance += chunk_size


def exist_in_batch_instance(job_name, batch_instance_path=BATCH_INSTANCE_PATH,
                            selected_batch_instance_path=SELECTED_BATCH_INSTANCE_PATH):
    """
    找出 batch_instance.csv 中与 batch_task.csv 中 task_name 一致的 jobs

    :param job_name:
    :param batch_instance_path:
    :param selected_batch_instance_path:
    :return: 将查找结果输出到文件中
    """

    if not os.path.exists(selected_batch_instance_path):
        columns = ['instance_name', 'task_name', 'job_name', 'task_type', 'status',
                   'start_time', 'end_time', 'machine_id', 'seq_no', 'total_seq_no',
                   'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        df = pd.DataFrame(columns=columns)
        df.to_csv(selected_batch_instance_path, index=False)

    ret, flag = False, False
    chunk_size, idx = 3, 0
    # 若能在 batch_instance.csv 中找到的最小的 job_name_num_in_batch_task, 仍然比 batch_task 中的大, 提前返回
    job_name_num_in_batch_instance = 0
    job_name_num_in_batch_task = int(re.findall(r"\d+\.?\d*", job_name)[0])
    for chunk in pd.read_csv(batch_instance_path, header=None, chunksize=chunk_size):
        # print(chunk)
        sampled_jobs_batch_instance = pd.DataFrame()  # 变量 sampled_jobs_batch_instance 用于保存采样出来的 sampled_jobs_batch_instance

        # 在 batch_instance.csv 中查找名为 job_name 的 job, 将其挑选出来, 保存到文件中
        i = 0
        while i < chunk_size:
            instance_nums = 0  # 计数：所有 job name 相同的

            print(job_name + " - " + chunk.loc[idx + i + instance_nums, 2])

            # 若 job_name_num_in_batch_task < job_name_num_in_batch_instance 可以提前终止
            job_name_num_in_batch_instance = int(re.findall(r"\d+\.?\d*", chunk.loc[idx + i + instance_nums, 2])[0])
            if job_name_num_in_batch_task < job_name_num_in_batch_instance:
                return ret, job_name_num_in_batch_instance

            while job_name == chunk.loc[idx + i + instance_nums, 2]:
                instance_nums += 1
            if instance_nums == 0:  # 当前条目不符，查找下一个条目
                i += 1
            else:  # 找到了条件相符的 job, 将其保存下来
                sampled_jobs_batch_instance = pd.concat(
                    [sampled_jobs_batch_instance, chunk.loc[idx + i: idx + i + instance_nums - 1].copy()], axis=0)
                i += instance_nums
                idx += i
                job_name_num_in_batch_instance = job_name_num_in_batch_task
                # if i < chunk_size:
                #     break

        if not sampled_jobs_batch_instance.empty:
            with open(selected_batch_instance_path, 'a') as f:
                sampled_jobs_batch_instance.to_csv(f, header=False)
            ret = True  # 如果 batch_instance 中有和 batch_task 一致的 task_name, 返回 True
        else:
            idx = idx + chunk_size

    return ret, job_name_num_in_batch_instance


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
    # exist_in_batch_instance("j_1")
