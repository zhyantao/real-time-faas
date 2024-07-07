"""工具类"""
import pprint
import random

from models.utils.parameters import args


def get_one_machine(machines, start_idx=0):
    """
    从当前 start_idx 开始，获取一个 machine，并返回下一个 machine 的 idx
    :return: 返回从 start_idx 开始的第一个 machine，以及下一个 machine 的起始 idx
    """
    machine_id = machines.loc[start_idx, 'machine_id']
    rows = machines.shape[0]
    machine_nums = 0
    while (start_idx + machine_nums < rows) and (machines.loc[start_idx + machine_nums, 'machine_id'] == machine_id):
        machine_nums += 1

    machine = machines.loc[start_idx: start_idx + machine_nums - 1].copy()
    next_idx = start_idx + machine_nums
    return machine, next_idx


def get_one_job(jobs, start_idx=0):
    """
    从当前 start_idx 开始，获取一个 job，并返回下一个 job 的 idx.

    :param jobs: jobs 列表，DataFrame 格式
    :param start_idx: 指明需要从哪里开始获取，没有指定默认是 0
    :return: 返回从 start_idx 开始的第一个 job，以及下一个 job 的起始 idx
    """
    job_name = jobs.loc[start_idx, 'job_name']
    rows = jobs.shape[0]
    task_nums = 0
    while (start_idx + task_nums < rows) and (jobs.loc[start_idx + task_nums, 'job_name'] == job_name):
        task_nums = task_nums + 1
    job = jobs.loc[start_idx: start_idx + task_nums - 1].copy()
    next_idx = start_idx + task_nums
    return job, next_idx


def reverse_dict(d):
    """ Reverses direction of dependence dict.
    e.g.:
    d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    reverse_dict(d) = {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
    """
    result = {}
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key,)
    return result


def generate_random_numbers(count):
    """生成 count 个不重复的随机数，总和为 1"""
    numbers = random.sample(range(1, 100), count)
    total = sum(numbers)
    return [round(number / total, 2) for number in numbers]


# DPE 和 HEFT 算法
def get_simple_paths(G):
    """
    获取任意两个 server 之间的简单路径
    """
    simple_paths = []
    for i in range(args.n_nodes):
        paths_from_i = []
        for j in range(args.n_nodes):
            node = i
            node_dst = j
            paths_ij, path_ij = [], []
            path_nodes_ij = set()
            go_forward(node, node_dst, paths_ij, path_ij, path_nodes_ij, G)
            paths_from_i.append(paths_ij)
        simple_paths.append(paths_from_i)
    return simple_paths


def go_forward(node, node_dst, paths_ij, path_ij, path_nodes_ij, G):
    """
    该算法用于寻找 server_i 和 server_j 之间所有的简单路径
    """
    if node == node_dst:
        path_ij.append(node)
        paths_ij.append(path_ij[:])
        path_ij.pop()
    else:
        path_ij.append(node)
        path_nodes_ij.add(node)  # 将 node 添加到集合 path_nodes_ij 中
        for i in range(args.n_nodes):
            if G[node][i] and (i not in path_nodes_ij):
                go_forward(i, node_dst, paths_ij, path_ij, path_nodes_ij, G)
        path_ij.pop()
        path_nodes_ij.discard(node)  # 从集合 path_nodes_ij 中移除 node


def print_simple_paths(simple_paths):
    """
    打印所有的简单路径
    :param simple_paths:
    :return:
    """
    print('\n====> All simple paths between any two server <====')
    pprint.pprint(simple_paths)


def print_simple_path(simple_paths, i, j):
    """
    打印指定 server 之间的简单路径
    :param simple_paths:
    :param i:
    :param j:
    :return:
    """
    print('\n====> All simple paths between server %d and %d <====' % (i + 1, j + 1))
    print('\nFrom server %d to server %d:' % (i + 1, j + 1))
    pprint.pprint(simple_paths[i][j])
    print('\nFrom server %d to server %d:' % (j + 1, i + 1))
    pprint.pprint(simple_paths[j][i])


def get_ratio(simple_paths, bw):
    """
    对每条简单路径上的每条连接，计算带宽的倒数和。然后获得根据结果对数据流分配比例。
    """
    reciprocals_list = []
    proportions_list = []

    for i in range(args.n_nodes):
        reciprocals = []
        proportions = []
        for j in range(args.n_nodes):
            paths = simple_paths[i][j]
            paths_len = len(paths)
            reciprocal_sum_list = []
            # 统计从 server_i 到 server_j 的所有简单路径的带宽的倒数之和
            if i != j:
                for k in range(paths_len):
                    path = paths[k]
                    path_len = len(path)
                    reciprocal_sum = 0
                    for l in range(path_len - 1):
                        # 计算一条简单路径上的带宽倒数之和
                        reciprocal_sum = reciprocal_sum + 1 / bw[path[l], path[l + 1]]
                    # 一条简单路径的计算结果保存一次
                    reciprocal_sum_list.append(reciprocal_sum)
            # 从 server_i 到 server_j 的所有简单路径以列表的形式保存了下来
            reciprocals.append(reciprocal_sum_list)

            if len(reciprocal_sum_list) > 0:
                # the source node and the destination node are different nodes
                # 核心代码：数据流分配方式是以带宽为依据的
                proportions.append(1. / sum(reciprocal_sum_list[0] / reciprocal_sum_list))
            else:
                proportions.append(-1)

        # 添加 server_i 到其他所有 server_j 的数据
        reciprocals_list.append(reciprocals)
        proportions_list.append(proportions)

    return reciprocals_list, proportions_list


def print_scenario(G, bw, pp):
    print('\nThe connected graph of nodes (represented by adjacent matrix):')
    pprint.pprint(G)
    print('\n====> throughput of each link <====')
    pprint.pprint(bw)  # 作者将 node_i 和 node_i 的吞吐量设置为 -1 不合理，应该是 MAX_VALUE
    print('\n====> processing power of edge server <====')
    pprint.pprint(pp)
