from spark_env.executor import Executor


class MultiResExecutor(Executor):

    def __init__(self, idx, exec_type, cpu, mem, ):
        Executor.__init__(self, idx)
        self.idx = idx
        self.type = exec_type
        self.cpu = cpu
        self.mem = mem

    def fit(self, cpu, mem):
        return self.cpu >= cpu and self.mem >= mem
