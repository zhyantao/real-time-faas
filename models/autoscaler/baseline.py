class Baseline:
    def __init__(self, job, base_index):
        self.job = job
        self.base_index = base_index

    def get_time_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，不同的 instance 几乎同时启动，所以，总时间成本为每个 instance 的时间成本之和
        """
        df = self.job
        df_rows = df.shape[0]

        start_time = df.loc[:, 'start_time']
        end_time = df.loc[:, 'end_time']

        total, miss_rows = 0, 0
        for idx in range(df_rows):
            st = start_time[self.base_index + idx]
            et = end_time[self.base_index + idx]
            if st < et:
                total += et - st
            else:
                miss_rows += 1  # 丢失或错误数据

        # 解决除 0 异常
        if df_rows == miss_rows:
            return 0

        avg = total / (df_rows - miss_rows)

        return avg

    def get_cpu_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的 CPU 消耗为每个 instance 的 CPU 消耗之和
        """
        df = self.job
        df_rows = df.shape[0]

        cpu_avg = df.loc[:, 'cpu_avg']

        total, miss_rows = 0, 0
        for idx in range(df_rows):
            ca = cpu_avg[self.base_index + idx]
            if ca >= 0:
                total += ca
            else:
                miss_rows += 1

        # 解决除 0 异常
        if df_rows == miss_rows:
            return 0

        avg = total / (df_rows - miss_rows)

        return avg

    def get_mem_cost(self):
        """
        由于默认情况下 instance 运行在不同的机器上，所以，总的内存消耗为每个 instance 的内存消耗之和
        """
        df = self.job
        df_rows = df.shape[0]

        mem_avg = df.loc[:, 'mem_avg']

        total, miss_rows = 0, 0
        for idx in range(df_rows):
            ma = mem_avg[self.base_index + idx]
            if 0 <= ma <= 100:
                total += ma
            else:
                miss_rows += 1

        # 解决除 0 异常
        if df_rows == miss_rows:
            return 0

        avg = total / (df_rows - miss_rows)

        return avg
