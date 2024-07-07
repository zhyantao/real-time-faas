import numpy as np

from models.scheduler.env import Environment
from models.scheduler.node import Node
from models.scheduler.task import Task
from models.utils.parameters import args


def load_env():
    """初始化环境和调度器（调度器也就是强化学习网络）"""
    tasks = load_tasks('j_82634')
    task_generator = (t for t in tasks)
    ndoes = load_nodes()

    # 生成状态空间
    # 从这里生成状态空间，Environment( nodes, queue, backlog, tasks)
    # 可以观察到状态空间共分成四部分，分别代表了运行节点、工作队列、积压队列、任务
    env = Environment(ndoes, args.queue_size, args.backlog_size, task_generator)
    env.timestep()

    return env


def load_tasks(job_name):
    """加载 tasks"""
    adj_matrix = np.load(args.task_depend_prefix + job_name + '_adj_matrix.npy')
    data_size = np.load(args.task_depend_prefix + job_name + '_data_size.npy')
    required_cpu = np.load(args.task_depend_prefix + job_name + '_required_cpu.npy')
    required_mem = np.load(args.task_depend_prefix + job_name + '_required_mem.npy')
    duraton = np.load(args.task_depend_prefix + job_name + '_duration.npy')

    tasks = []
    for i in range(adj_matrix.shape[0]):
        tasks.append(Task([required_cpu[i], required_mem[i], data_size[i]], duraton[i], 'task' + str(i)))

    return tasks


def load_nodes():
    """加载 nodes"""
    udg_name = str(args.n_nodes) + '_nodes_' + str(args.n_max_connections) + '_connections'
    npy_file_name_prefix = args.node_connect_prefix + udg_name
    adj_matrix = np.load(npy_file_name_prefix + '_adj_matrix.npy')
    processing_power = np.load(npy_file_name_prefix + '_processing_power.npy')

    nodes = []
    for i in range(adj_matrix.shape[0]):
        n_resource_types = 3  # 共 3 种资源类型：data_size、cpu、内存
        nodes.append(Node(processing_power, 100, 'node' + str(i)))  # 这里处理的不太好

    return nodes


if __name__ == '__main__':
    env = load_env()
    print(env)
