from spark_env.task import Task


class MultiResTask(Task):
    def __init__(self, idx, cpu, mem, rough_duration, wall_time):
        Task.__init__(self, idx, rough_duration, wall_time)

    def fit(self, executor):
        return executor.fit(self.cpu, self.mem)
