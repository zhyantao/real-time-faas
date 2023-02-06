import numpy as np
import pandas as pd

from models.utils.parameters import *


class Baseline:
    def __int__(self):
        pass

    def get_time_cost(self, file_path_alloc_sorted=FILE_PATH_ALLOC_SORTED):
        """
        计算 Alibaba Trace Data 的默认计算耗时
        """
        df = pd.read_csv(file_path_alloc_sorted)
        df_len = df.shape[0]
        min_start_time = np.inf
        max_end_time = 0
        for idx in range(df_len):
            if df.iat[idx, 5] < min_start_time:
                min_start_time = df.iat[idx, 5]
                # print('min: ' + str(min_start_time))
            if df.iat[idx, 6] > max_end_time:
                max_end_time = df.iat[idx, 6]
                # print('max: ' + str(max_end_time))
        return max_end_time - min_start_time

    def get_cpu_cost(self, file_path_alloc_sorted=FILE_PATH_ALLOC_SORTED):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的 CPU 消耗为每个 instance 的 CPU 消耗之和
        """
        df = pd.read_csv(file_path_alloc_sorted)
        df_len = df.shape[0]
        sum = 0
        for idx in range(df_len):
            sum += df.iat[idx, 10]
        return sum

    def get_mem_cost(self, file_path_alloc_sorted=FILE_PATH_ALLOC_SORTED):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的内存消耗为每个 instance 的内存消耗之和
        """
        df = pd.read_csv(file_path_alloc_sorted)
        df_len = df.shape[0]
        sum = 0
        for idx in range(df_len):
            sum += df.iat[idx, 12]
        return sum
