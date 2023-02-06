import pandas as pd

from models.utils.parameters import *


class Baseline:
    def __int__(self, batch_task_path=SELECTED_BATCH_TASK_PATH, batch_instance_path = SELECTED_BATCH_INSTANCE_PATH):
        self.batch_task = pd.read_csv(batch_task_path)
        self.batch_instance = pd.read_csv(batch_instance_path)
        pass

    def get_time_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，不同的 instance 几乎同时启动，所以，总时间成本为每个 instance 的时间成本之和
        """
        df = self.batch_task
        df_len = df.shape[0]
        sum = 0
        for idx in range(df_len):
            sum += df.iat[idx, 6] - df.iat[idx, 5]
        return sum

    def get_cpu_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的 CPU 消耗为每个 instance 的 CPU 消耗之和
        """
        df = self.batch_instance
        df_len = df.shape[0]
        sum = 0
        for idx in range(df_len):
            sum += df.iat[idx, 10]
        return sum

    def get_mem_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的内存消耗为每个 instance 的内存消耗之和
        """
        df = self.batch_instance
        df_len = df.shape[0]
        sum = 0
        for idx in range(df_len):
            sum += df.iat[idx, 12]
        return sum
