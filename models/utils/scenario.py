"""
Generate the edge computing scenario and set functions. We simply set sigma_{ij} as zero.

生成边缘计算场景和函数集合。这里仅仅简单地把 sigma_{ij} 设置为 0。
"""
import pprint
import random

import numpy as np
import yaml

with open('E:/Workshop/real-time-faas/configs/parameter.yaml', 'r') as f:
    para = yaml.load(f, Loader=yaml.FullLoader)


def generate_scenario():
    """
    Generate the edge computing scenario, i.e., generate a connected graph of edge servers,
    including the
        connectivity,
        processing power of each server,
        bandwidth of each physical link.
    """
    # step 1: generate a connected graph
    # initialize 以邻接矩阵的方式表示图
    # G_{ij} = 1 表示 node_i 和 node_j 之间可以通信，否则不能
    # D_{ij} = MAX_VALUE 表示 node_i 和 node_j 无法通信，否则表示最短通信距离
    G = np.zeros((para.get("cpu_nums"), para.get("cpu_nums")))
    D = np.ones((para.get("cpu_nums"), para.get("cpu_nums"))) * eval(para.get("max_value"))

    is_connected = False
    for i in range(para.get("cpu_nums")):
        G[i, i] = 1
        D[i, i] = 0

    while not is_connected:
        for i in range(para.get("cpu_nums")):
            # randomly connect i and at most 'DENSITY' other servers
            # 将 node_i 随机连接其他 node，最多连接 DENSITY 个其他 node
            conn_node_num = random.randint(0, para.get("density"))
            for j in range(conn_node_num):
                k = random.randint(0, para.get("cpu_nums") - 1)
                G[i, k], G[k, i] = 1, 1

        for i in range(para.get("cpu_nums")):
            # 将 jobs 的连通性信息复制到 D
            for j in range(para.get("cpu_nums")):
                if G[i, j] == 1:
                    D[i, j], D[j, i] = 1, 1

        for i in range(para.get("cpu_nums")):
            for j in range(para.get("cpu_nums")):
                for k in range(para.get("cpu_nums")):
                    # 若 node_j 到 node_k 需要经过 node_i，且经过 node_i 可以缩短距离
                    # 那么，将 node_j 到 node_k 的距离更新为更短的距离
                    if D[j, k] > D[j, i] + D[i, k]:
                        D[j, k] = D[j, i] + D[i, k]

        is_continue = False
        for i in range(para.get("cpu_nums")):
            for j in range(para.get("cpu_nums")):
                # 如果 node_{ij} == MAX_VALUE，标识这两个节点之间的的距离非常大
                # MAX_VALUE 是一个标志数字，不具备实际意义，只用来标识连接性
                if D[i, j] == para.get("max_value"):
                    # the graph is not a connected graph
                    is_continue = True
                    break
        # 判断最后一个节点是否具有连接性
        if not is_continue:
            is_connected = True

    # step 2: set the bandwidth
    bw = np.ones((para.get("cpu_nums"), para.get("cpu_nums"))) * -1
    for i in range(para.get("cpu_nums")):
        j = 0
        while j < i:
            if G[i, j] == 1:
                # 对能通信的 node_i 和 node_j 设置一个随机的带宽
                b = random.randint(para.get("bw_lower"), para.get("bw_upper"))
                bw[i, j], bw[j, i] = b, b
            j = j + 1

    # step 3: set the processing power
    # 生成长度为 para.get("cpu_nums") 的随机数组，元素的取值范围为 [para.get("pp_lower"), para.get("pp_upper")]
    pp = np.random.randint(para.get("pp_lower"), para.get("pp_upper"), para.get("cpu_nums"))

    return G, bw, pp


def print_scenario(G, bw, pp):
    print('\nThe connected graph of edge servers (represented by adjacent matrix):')
    pprint.pprint(G)
    print('\n====> throughput of each link <====')
    pprint.pprint(bw)  # 作者将 node_i 和 node_i 的吞吐量设置为 -1 不合理，应该是 MAX_VALUE
    print('\n====> processing power of edge server <====')
    pprint.pprint(pp)


def go_forward(node, node_dst, paths_ij, path_ij, path_nodes_ij, G):
    """
    The recursive algorithm (OSM) to find all the simple paths between any two node i and j.

    该算法（OSM）用于寻找 server_i 和 server_j 之间所有的简单路径
    """
    if node == node_dst:
        path_ij.append(node)
        paths_ij.append(path_ij[:])
        path_ij.pop()
    else:
        path_ij.append(node)
        path_nodes_ij.add(node)  # 将 node 添加到集合 path_nodes_ij 中
        for i in range(para.get("cpu_nums")):
            if G[node][i] and (i not in path_nodes_ij):
                go_forward(i, node_dst, paths_ij, path_ij, path_nodes_ij, G)
        path_ij.pop()
        path_nodes_ij.discard(node)  # 从集合 path_nodes_ij 中移除 node


def get_simple_paths(G):
    """
    Get all the simple paths between any two edge servers. Call the subroutine go_forward().

    获取任意两个 server 之间的简单路径
    """
    simple_paths = []
    for i in range(para.get("cpu_nums")):
        paths_from_i = []
        for j in range(para.get("cpu_nums")):
            node = i
            node_dst = j
            paths_ij, path_ij = [], []
            path_nodes_ij = set()
            go_forward(node, node_dst, paths_ij, path_ij, path_nodes_ij, G)
            paths_from_i.append(paths_ij)
        simple_paths.append(paths_from_i)
    return simple_paths


def print_simple_paths(simple_paths):
    print('\n====> All simple paths between any two server <====')
    pprint.pprint(simple_paths)


def print_simple_path(simple_paths, i, j):
    print('\n====> All simple paths between server %d and %d <====' % (i + 1, j + 1))
    print('\nFrom server %d to server %d:' % (i + 1, j + 1))
    pprint.pprint(simple_paths[i][j])
    print('\nFrom server %d to server %d:' % (j + 1, i + 1))
    pprint.pprint(simple_paths[j][i])


def get_ratio(simple_paths, bw):
    """
    Calculate the sum of the reciprocal of bandwidth of each link for every simple path.
    Then, get the proportion of data stream size which routes through the first simple path between any two nodes.
    (The first simple path between server i and j is stored in simple_paths[i][j][0].)

    - 对每条简单路径上的每条连接，计算带宽的倒数和。
    - 然后获得根据结果对数据流分配比例。
    """
    reciprocals_list = []
    proportions_list = []

    for i in range(para.get("cpu_nums")):
        reciprocals = []
        proportions = []
        for j in range(para.get("cpu_nums")):
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


def set_funcs():
    """
    设置处理单元的处理能力和需要处理的数据大小，这两个变量被所有的 jobs 共用。
    """
    # set the processing power required
    pp_required = np.random.randint(para.get("pp_required_lower"), para.get("pp_required_upper"),
                                    (para.get("max_task_nums")))
    data_stream = np.random.randint(para.get("data_stream_size_lower"), para.get("data_stream_size_upper"),
                                    (para.get("max_task_nums")))
    return pp_required, data_stream
