"""
This script defines all the parameters used.
"""
import os


# csv file path
DIR_PATH = 'E:/Workshop/real-time-faas/dataset/'
DATASET_PATH = os.path.join(DIR_PATH, 'batch_task.csv')
SELECTED_DAG_PATH = os.path.join(DIR_PATH, 'selected_DAGs.csv')
SORTED_DAG_PATH = os.path.join(DIR_PATH, 'topological_order.csv')
TEST_DAG_PATH = os.path.join(DIR_PATH, 'test.csv')

BATCH_TASK_PATH = os.path.join(DIR_PATH, "batch_task.csv")
SELECTED_BATCH_TASK_PATH = os.path.join(DIR_PATH, "selected_batch_task.csv")
BATCH_TASK_TOPOLOGICAL_ORDER_PATH = os.path.join(DIR_PATH, 'batch_task_topological_order.csv')

BATCH_INSTANCE_PATH = os.path.join(DIR_PATH, "batch_instance.csv")
SELECTED_BATCH_INSTANCE_PATH = os.path.join(DIR_PATH, "selected_batch_instance.csv")
BATCH_INSTANCE_TOPOLOGICAL_ORDER_PATH = os.path.join(DIR_PATH, 'batch_instance_topological_order.csv')

MAX_VALUE = 9e+4
REQUIRED_NUM = [200, 800, 600, 400, 119]
MAX_FUNC_NUM = 250


class Parameter:
    def __init__(self):
        # edge computing environment settings
        self.__server_num = 4
        self.__n_pairs = self.__server_num * (self.__server_num - 1)
        # density is used to adjust the connectivity of the graph
        self.__density = 10
        # bandwidth generation scope
        self.__bw_lower, self.__bw_upper = 30, 70
        # processing power scope
        self.__pp_lower, self.__pp_upper = 7, 14

        # DAG settings (processing power required by each function, the data stream size of each link)
        self.__pp_required_lower, self.__pp_required_upper = 1, 2  # 设置每个函数需要的 processing power 上下限
        self.__data_stream_size_lower, self.__data_stream_size_upper = 1, 10
        self.__max_func_num = MAX_FUNC_NUM

    def get_server_num(self):
        return self.__server_num

    def get_n_pairs(self):
        return self.__n_pairs

    def get_density(self):
        return self.__density

    def get_bw_lower(self):
        return self.__bw_lower

    def get_bw_upper(self):
        return self.__bw_upper

    def get_pp_lower(self):
        return self.__pp_lower

    def get_pp_upper(self):
        return self.__pp_upper

    def get_pp_required_lower(self):
        return self.__pp_required_lower

    def get_pp_required_upper(self):
        return self.__pp_required_upper

    def get_data_stream_size_lower(self):
        return self.__data_stream_size_lower

    def get_max_func_num(self):
        return self.__max_func_num

    def get_data_stream_size_upper(self):
        return self.__data_stream_size_upper
