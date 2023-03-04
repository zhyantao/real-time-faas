from collections import OrderedDict

from param import *
from spark_env.executor_commit import ExecutorCommit


class MultiResExecutorCommit(ExecutorCommit):
    def __init__(self):
        ExecutorCommit.__init__(self)
        self.num_exec_group = len(args.exec_group_num)
        self.commit = [{} for _ in range(self.num_exec_group)]

    def add_job(self, job_dag):
        # add commit entry to the map
        for i in range(self.num_exec_group):
            self.commit[i][job_dag] = OrderedDict()
        for node in job_dag.nodes:
            for i in range(self.num_exec_group):
                self.commit[i][node] = OrderedDict()
            self.node_commit[node] = 0
            for i in range(self.num_exec_group):
                self.backward_map[(i, node)] = set()

    def add(self, source, node, exec_type, amount):
        # source can be node or job
        # node: executors continuously free up
        # job: free executors

        # add foward connection
        if node not in self.commit[exec_type][source]:
            self.commit[exec_type][source][node] = 0
        # add node commit
        self.commit[exec_type][source][node] += amount
        # add to record of total commit on node
        self.node_commit[node] += amount
        # add backward connection
        self.backward_map[(exec_type, node)].add(source)

    def get_len(self, exec_type, source):
        return len(self.commit[exec_type][source])

    def pop(self, exec_type, source):
        # implicitly assert source in self.commit
        # implicitly assert len(self.commit[source]) > 0

        # find the node in the map
        node = next(iter(self.commit[exec_type][source]))

        # deduct one commitment
        self.commit[exec_type][source][node] -= 1
        self.node_commit[node] -= 1
        assert self.commit[exec_type][source][node] >= 0
        assert self.node_commit[node] >= 0

        # remove commitment on job if exhausted
        if self.commit[exec_type][source][node] == 0:
            del self.commit[exec_type][source][node]
            self.backward_map[(exec_type, node)].remove(source)

        return node

    def remove_job(self, job_dag):
        # when removing jobs, the commiment should be all satisfied
        for i in range(self.num_exec_group):
            assert len(self.commit[i][job_dag]) == 0
        del self.commit[i][job_dag]

        # clean up commitment to the job
        for node in job_dag.nodes:
            for i in range(self.num_exec_group):
                # the executors should all move out
                for n in self.commit[i][node]:
                    self.backward_map[(i, n)].remove(node)
                del self.commit[i][node]

            for i in range(self.num_exec_group):
                for source in self.backward_map[(i, node)]:
                    # remove forward link
                    del self.commit[i][source][node]
                # remove backward link
                del self.backward_map[(i, node)]
            # remove node commit records
            del self.node_commit[node]

    def reset(self):
        ExecutorCommit.reset(self)
        self.backward_map = {}
        self.commit = [{} for _ in range(self.num_exec_group)]
        for i in range(self.num_exec_group):
            self.commit[i][None] = OrderedDict()
            self.backward_map[(i, None)] = set()
