"""有向图生成器：数据来源于随机生成的序列"""
import random

import networkx as nx
import numpy as np

from models.utils.parameters import args


class UDG:
    def __init__(self):
        pass

    def generate_udg_from_random(self, n_nodes, n_max_connections, bandwidth_lower, bandwidth_upper):
        """
        根据指定的 n_nodes 和 n_max_connections 生成节点拓扑结构。
        注：最大连接数为 1 时，会形成多个连通子图。最大连接数为 0 表示每个节点都是单独的。
        """

        # 每个节点的最大连接数量不能超过节点的总数
        assert 0 < n_max_connections < n_nodes

        # 根据最大连接数量，初始化拓扑结构
        connection_set = [set() for _ in range(n_nodes)]
        for i in range(n_nodes):
            n_nodes_already_connected = len(connection_set[i])
            n_nodes_all_connections = np.random.randint(1, n_max_connections + 1)
            while n_nodes_already_connected < n_nodes_all_connections:
                new_node_number = random.randint(0, n_nodes - 1)

                # 新连接不能跟自己相连
                if new_node_number == i:
                    continue

                # 如果另外一个连接已经达到了最大值，生成的此连接无效，需要重新生成
                if len(connection_set[new_node_number]) >= n_max_connections:
                    continue

                # 有效连接，可以继续
                connection_set[i].add(new_node_number)
                connection_set[new_node_number].add(i)
                n_nodes_already_connected = len(connection_set[i])

        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(i)

        for i in range(n_nodes):
            for j in connection_set[i]:
                G.add_edge(i, j, weight=random.randint(bandwidth_lower, bandwidth_upper))
        adj = np.asarray(nx.to_numpy_array(G))
        udg_name = str(n_nodes) + ' nodes ' + str(n_max_connections) + ' connections'
        npy_file_name = udg_name.replace(' ', '_') + '_adj_matrix'
        np.save(args.node_connect_prefix + npy_file_name, adj)
        return G, udg_name
