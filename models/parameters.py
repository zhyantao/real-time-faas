"""
This script defines all the parameters used.
"""
import os


# csv file path
DIR_PATH = '../'
DATASET_PATH = os.path.join(DIR_PATH, 'dataset/batch_task.csv')
SELECTED_DAG_PATH = os.path.join(DIR_PATH, 'dataset/selected_DAGs.csv')
SORTED_DAG_PATH = os.path.join(DIR_PATH, 'dataset/topological_order.csv')
TEST_DAG_PATH = os.path.join(DIR_PATH, 'dataset/test.csv')

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
        self.__pp_required_lower, self.__pp_required_upper = 1, 2
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
