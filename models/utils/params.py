"""输入参数控制"""
import argparse

ROOT_PATH = 'E:/Workshop/real-time-faas/'
DATASET_PATH = ROOT_PATH + 'dataset/'
RESULTS_PATH = ROOT_PATH + 'results/'

parser = argparse.ArgumentParser(description='REAL_TIME_FAAS')

# -- Dataset --
parser.add_argument('--selected_container_usage_path', type=str, default=DATASET_PATH + 'selected_container_usage.csv',
                    help='从 container_usage.csv 中选出的数据')
parser.add_argument('--container_usage_path', type=str, default=DATASET_PATH + 'container_usage.csv',
                    help='从 container_usage.csv 中选出的数据')
parser.add_argument('--total_machines', type=int, default=100, help='总共的 node 数量')
parser.add_argument('--total_jobs', type=int, default=100, help='总共需要处理的 job 数量')
parser.add_argument('--batch_task_path', type=str, default=DATASET_PATH + 'batch_task.csv', help='原始数据 batch_task.csv')
parser.add_argument('--selected_batch_task_path', type=str, default=DATASET_PATH + 'selected_batch_task.csv',
                    help='batch_task.csv 采样后的数据')
parser.add_argument('--batch_task_topological_order_path', type=str,
                    default=DATASET_PATH + 'batch_task_topological_order.csv',
                    help='selected_batch_task.csv 拓扑排序后的数据')
parser.add_argument('--batch_instance_path', type=str, default=DATASET_PATH + 'batch_instance.csv',
                    help='原始数据 batch_instance.csv')
parser.add_argument('--selected_batch_instance_path', type=str, default=DATASET_PATH + 'selected_batch_instance.csv',
                    help='batch_instance.csv 采样后的数据')
parser.add_argument('--batch_instance_topological_order_path', type=str,
                    default=DATASET_PATH + 'batch_instance_topological_order.csv',
                    help='selected_batch_instance.csv 拓扑排序后的数据')
parser.add_argument('--result_saving_path', type=str, default=RESULTS_PATH, help='保存实验结果的路径')

# job settings
parser.add_argument('--max_value', type=int, default=9e+4, help='最大值')
parser.add_argument('--max_task_nums', type=int, default=250, help='最大任务数量')
parser.add_argument('--data_stream_size_upper', type=int, default=10, help='传输数据大小的上限')
parser.add_argument('--data_stream_size_lower', type=int, default=1, help='传输数据大小的下限')
parser.add_argument('--pp_required_upper', type=int, default=2, help='需要的处理器数量上限')
parser.add_argument('--pp_required_lower', type=int, default=1, help='需要的处理器数量下限')
parser.add_argument('--pp_upper', type=int, default=14, help='已有的处理器数量上限')
parser.add_argument('--pp_lower', type=int, default=7, help='已有的处理器数量下限')
parser.add_argument('--bw_upper', type=int, default=70, help='已有的带宽总量上限')
parser.add_argument('--bw_lower', type=int, default=30, help='已有的带宽总量下限')
parser.add_argument('--density', type=int, default=5, help='节点的最大连接密度')
parser.add_argument('--n_pairs', type=int, default=12, help='？？？')
parser.add_argument('--cpu_nums', type=int, default=10, help='处理器数量')

args = parser.parse_args(args=[])
