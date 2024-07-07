import argparse

ROOT_PATH = './'
RAW_DATASET_PATH = './dataset/'
SELECTED_DATASET_PATH = ROOT_PATH + 'dataset/'
RESULT_PATH = ROOT_PATH + 'results/'

parser = argparse.ArgumentParser(description='real-time-faas')

# RAW DATASET
parser.add_argument('--batch_task_path', type=str,
                    default=RAW_DATASET_PATH + 'batch_task.csv',
                    help='原始数据 batch_task.csv')
parser.add_argument('--container_usage_path', type=str,
                    default=RAW_DATASET_PATH + 'container_usage.csv',
                    help='记录资源利用率的文件')
parser.add_argument('--batch_instance_path', type=str,
                    default=RAW_DATASET_PATH + 'batch_instance.csv',
                    help='原始数据 batch_instance.csv')

# SELECTED DATASET
parser.add_argument('--selected_batch_task_path', type=str,
                    default=SELECTED_DATASET_PATH + 'selected_batch_task.csv',
                    help='batch_task.csv 采样后的数据')
parser.add_argument('--selected_container_usage_path', type=str,
                    default=SELECTED_DATASET_PATH + 'selected_container_usage.csv',
                    help='从 container_usage.csv 中选出的数据')
parser.add_argument('--selected_batch_instance_path', type=str,
                    default=SELECTED_DATASET_PATH + 'selected_batch_instance.csv',
                    help='batch_instance.csv 采样后的数据')

# INPUT DATASET
parser.add_argument('--task_depend_prefix', type=str,
                    default=SELECTED_DATASET_PATH + 'task_depend_',
                    help='任务调度需要的任务间依赖信息以及任务间的转移概率信息')
parser.add_argument('--node_connect_prefix', type=str,
                    default=SELECTED_DATASET_PATH + 'node_connect_',
                    help='任务调度需要的节点间的连通性信息和节点间的带宽大小')
parser.add_argument('--batch_task_topological_order_path', type=str,
                    default=SELECTED_DATASET_PATH + 'batch_task_topological_order.csv',
                    help='selected_batch_task.csv 拓扑排序后的数据')

# RESULT
parser.add_argument('--result_saving_path', type=str,
                    default=RESULT_PATH,
                    help='保存实验结果的路径')

# Setting for preprocessing
parser.add_argument('--total_machines', type=int,
                    default=100,
                    help='总共需要从 container_usage.csv 中提取的 node 数量')
parser.add_argument('--total_jobs', type=int,
                    default=100,
                    help='总共需要从 batch_task.csv 中采样的 job 数量')

# Setting for node connections, node processing power, bandwidth
parser.add_argument('--n_nodes', type=int,
                    default=10,
                    help='总共的节点数量')
parser.add_argument('--n_max_connections', type=int,
                    default=5,
                    help='每个节点最大能够连接的其他节点的数量')
parser.add_argument('--bandwidth_lower', type=int,
                    default=20,
                    help='节点之间的带宽下限')
parser.add_argument('--bandwidth_upper', type=int,
                    default=60,
                    help='节点之间的带宽上限')
parser.add_argument('--processing_power_lower', type=int,
                    default=7,
                    help='节点每秒钟能够处理的数据下限（MB/s）')
parser.add_argument('--processing_power_upper', type=int,
                    default=14,
                    help='节点每秒钟能够处理的数据上限（MB/s）')

# Setting for resource requirements of tasks (only used for simulation)
parser.add_argument('--required_cpu_lower', type=int,
                    default=10,
                    help='需要的 CPU 资源下限')
parser.add_argument('--required_cpu_upper', type=int,
                    default=30,
                    help='需要的 CPU 资源上限')
parser.add_argument('--required_mem_lower', type=int,
                    default=20,
                    help='需要的内存资源下限')
parser.add_argument('--required_mem_upper', type=int,
                    default=60,
                    help='需要的内存资源上限')
parser.add_argument('--data_size_lower', type=int,
                    default=5,
                    help='任务需要的处理的数据大小下限')
parser.add_argument('--data_size_upper', type=int,
                    default=10,
                    help='任务需要的处理的数据大小上限')
parser.add_argument('--duration_lower', type=int,
                    default=1,
                    help='任务需要占用的 CPU 时长下限')
parser.add_argument('--duration_upper', type=int,
                    default=3,
                    help='任务需要占用的 CPU 时长上限')

# Setting for GCN-DQN
parser.add_argument('--queue_size', type=int,
                    default=10,
                    help='可以被神经网络同时处理的任务个数')
parser.add_argument('--backlog_size', type=int,
                    default=20,
                    help='候选队列中可容纳的任务个数')

# Parse parameter
args = parser.parse_args(args=[])
