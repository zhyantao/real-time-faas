from spark_env.node import Node


class MultiResNode(Node):
    def __init__(self, idx, cpu, mem, tasks,
                 task_duration, wall_time, np_random):
        Node.__init__(self, idx, tasks, task_duration, wall_time, np_random)

        self.cpu = cpu
        self.mem = mem

    def fit(self, executor):
        return executor.fit(self.cpu, self.mem)
