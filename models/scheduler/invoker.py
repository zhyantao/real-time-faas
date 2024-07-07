import datetime

import torch

from models.scheduler.dpe import DPE
from models.scheduler.dqn import ConvNet
from models.scheduler.gcn import encoding
from models.scheduler.heft import HEFT
from models.scheduler.loader import load_env
from models.utils.dataset import get_topological_order, gen_heft_and_dpe_input, gen_gcn_dqn_input
from models.utils.parameters import args
from models.utils.tools import print_scenario, get_simple_paths, print_simple_paths, get_ratio


def run_heft():
    # 生成数据集
    get_topological_order()

    # 根据数据集生成 FaaS 场景
    G, bw, pp = gen_heft_and_dpe_input()  # 根据配置文件指定的 CPU 数量生成场景
    print_scenario(G, bw, pp)

    # 打印简单路径
    simple_paths = get_simple_paths(G)  # 简单路径不用变
    print_simple_paths(simple_paths)

    reciprocals_list, proportions_list = get_ratio(simple_paths, bw)  # 倒数关系不用变

    # 进行算法对比 HEFT
    heft = HEFT(G, bw, pp, simple_paths, reciprocals_list, proportions_list)
    start = datetime.datetime.now()
    cpu_task_mapping_list_all, task_deployment_all, makespan_avg_heft \
        = heft.get_response_time(sorted_job_path=args.batch_task_topological_order_path)

    print(task_deployment_all)

    end = datetime.datetime.now()
    print('Computer\'s running time:', (end - start).microseconds, 'microseconds')


def run_dpe():
    """
    调用 DPE 算法
    :return:
    """
    get_topological_order()

    # 根据数据集生成 FaaS 场景
    G, bw, pp = gen_heft_and_dpe_input()  # 根据配置文件指定的 CPU 数量生成场景
    print_scenario(G, bw, pp)

    # 打印简单路径
    simple_paths = get_simple_paths(G)  # 简单路径不用变
    print_simple_paths(simple_paths)

    reciprocals_list, proportions_list = get_ratio(simple_paths, bw)  # 倒数关系不用变

    # 根据上面的机器之间的连通性、带宽限制、流量分配要求初始化 DPE 需要使用的遍历
    dpe = DPE(G, bw, pp, simple_paths, reciprocals_list, proportions_list)

    start = datetime.datetime.now()

    # 调用算法
    cpu_earliest_finish_time_all_dpe, \
        task_deployment_all_dpe, \
        cpu_task_mapping_list_all_dpe, \
        task_start_time_all_dpe, \
        makespan_avg_dpe = dpe.get_response_time(sorted_job_path=args.batch_task_topological_order_path)

    print(task_deployment_all_dpe)

    # 结束计时
    end = datetime.datetime.now()
    print('Computer\'s running time:', (end - start).microseconds, 'microseconds')


def run_gcn_dqn():
    # 初始化状态空间
    task_adj_matrix, task_features, node_adj_matrix, node_features = gen_gcn_dqn_input()
    task_adj_matrix = torch.Tensor(task_adj_matrix)
    task_features = torch.Tensor(task_features)  # (n_samples, n_features)
    node_adj_matrix = torch.Tensor(node_adj_matrix)
    node_features = torch.Tensor(node_features)

    # features = np.concatenate((task_adj_matrix, task_features), axis=1)  # task 的特征拼接

    encoded_tasks = encoding(task_adj_matrix, task_features)
    encoded_nodes = encoding(node_adj_matrix, node_features)

    print('encoded_tasks.shape: ', encoded_tasks.shape)
    print('encoded_nodes.shape: ', encoded_nodes.shape)

    env = load_env()
    print(env)

    n_tasks = 30
    n_task_features = 4
    n_nodes = 10
    input_shape = (n_tasks, n_task_features, 1)
    output_shape = (n_nodes, 1)
    model = ConvNet(input_shape=(10, 400, 1), output_shape=31)
    x = torch.randn(31, 1, 10, 400)  # Conv2d 要求的输入形状为 [batch_size, channels, height, width]
    y = model(x)
    print(y)
