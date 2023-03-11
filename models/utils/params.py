import argparse

parser = argparse.ArgumentParser(description='REAL_TIME_FAAS')

# -- Dataset --
parser.add_argument('--selected_container_usage_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/selected_container_usage.csv',
                    help='从 container_usage.csv 中选出的数据')
parser.add_argument('--container_usage_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/container_usage.csv',
                    help='从 container_usage.csv 中选出的数据')
parser.add_argument('--total_machines',
                    type=int,
                    default=100,
                    help='总共的 machine 数量')
parser.add_argument('--total_jobs',
                    type=int,
                    default=100,
                    help='最大值')
parser.add_argument('--batch_task_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/batch_task.csv',
                    help='最大值')
parser.add_argument('--selected_batch_task_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/selected_batch_task.csv',
                    help='最大值')
parser.add_argument('--batch_task_topological_order_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/batch_task_topological_order.csv',
                    help='最大值')
parser.add_argument('--batch_instance_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/batch_instance.csv',
                    help='最大值')
parser.add_argument('--selected_batch_instance_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/selected_batch_instance.csv',
                    help='最大值')
parser.add_argument('--batch_instance_topological_order_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/dataset/batch_instance_topological_order.csv',
                    help='最大值')
parser.add_argument('--result_saving_path',
                    type=str,
                    default='E:/Workshop/real-time-faas/results',
                    help='保存实验结果的路径')

# job settings
parser.add_argument('--max_value',
                    type=int,
                    default=9e+4,
                    help='最大值')
parser.add_argument('--max_task_nums',
                    type=int,
                    default=250,
                    help='最大值')
parser.add_argument('--data_stream_size_upper',
                    type=int,
                    default=10,
                    help='最大值')
parser.add_argument('--data_stream_size_lower',
                    type=int,
                    default=1,
                    help='最大值')
parser.add_argument('--pp_required_upper',
                    type=int,
                    default=2,
                    help='最大值')
parser.add_argument('--pp_required_lower',
                    type=int,
                    default=1,
                    help='最大值')
parser.add_argument('--pp_upper',
                    type=int,
                    default=14,
                    help='最大值')
parser.add_argument('--pp_lower',
                    type=int,
                    default=7,
                    help='最大值')
parser.add_argument('--bw_upper',
                    type=int,
                    default=70,
                    help='最大值')
parser.add_argument('--bw_lower',
                    type=int,
                    default=30,
                    help='最大值')
parser.add_argument('--density',
                    type=int,
                    default=10,
                    help='最大值')
parser.add_argument('--n_pairs',
                    type=int,
                    default=12,
                    help='最大值')
parser.add_argument('--cpu_nums',
                    type=int,
                    default=4,
                    help='最大值')

args = parser.parse_args()
