import collections

import numpy as np


class Node(object):
    """Node 实体类"""

    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)
        self.timestep_counter = 0  # 计时器
        self.scheduled_tasks = []
        # state matrices: 0 表示已经被占用, 255 表示空闲
        self.state_matrices = [np.full((duration, resource), 255, dtype=np.uint8) for resource in resources]
        self._state_matrices_capacity = [[resource] * duration for resource in resources]

    def schedule(self, task):
        """
        将 task 分配到 node 上
        :param task:
        :return:
        """

        # 找到 CPU 的最早空闲时间
        start_time = self._satisfy(self._state_matrices_capacity, task.resources, task.duration)

        if start_time == -1:
            # 找不到满足要求的 node
            return False

        else:
            # 找到了
            for i in range(task.dimension):
                self._occupy(
                    self.state_matrices[i], self._state_matrices_capacity[i],
                    task.resources[i], task.duration, start_time)

            self.scheduled_tasks.append((task, self.timestep_counter + task.duration))

            return True

    def timestep(self):
        """
        前进到下一个时间

        :return:
        """

        self.timestep_counter += 1

        # 更新 state matrices
        for i in range(self.dimension):
            temp = np.delete(self.state_matrices[i], 0, axis=0)
            temp = np.append(temp, np.array([[255 for _ in range(temp.shape[1])]]), axis=0)
            self.state_matrices[i] = temp

        for i in range(self.dimension):
            self._state_matrices_capacity[i].pop(0)
            self._state_matrices_capacity[i].append(self.resources[i])

        # 删除已经完成的任务
        indices = []
        for i in range(len(self.scheduled_tasks)):
            if self.timestep_counter >= self.scheduled_tasks[i][1]:
                indices.append(i)

        for i in sorted(indices, reverse=True):
            del self.scheduled_tasks[i]

    def summary(self, bg_shape=None):
        """
        状态空间表示

        :param bg_shape:
        :return:
        """
        if self.dimension > 0:
            temp = self._expand(self.state_matrices[0], bg_shape)
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, self._expand(self.state_matrices[i], bg_shape)), axis=1)

            return temp

        else:
            return None

    def utilization(self):
        """
        计算利用率

        :return:
        """
        # print('node.py --> state_matrices: ')
        # for matrix in self.state_matrices:
        #     print(matrix)
        # print('node.py --> state_matrices end')
        return sum([collections.Counter(matrix.flatten()).get(0, 0)
                    for matrix in self.state_matrices]) / sum(self.resources) / self.duration

    def _satisfy(self, capacity_matrix, required_resources, required_duration):
        """
        找到能够满足资源需求的节点的最早空闲时间

        :param capacity_matrix:
        :param required_resources:
        :param required_duration:
        :return:
        """

        p1 = 0
        p2 = 0
        duration_bound = min([len(capacity) for capacity in capacity_matrix])

        while p1 < duration_bound and p2 < required_duration:
            if False in [capacity_matrix[i][p1] >= required_resources[i]
                         for i in range(len(required_resources))]:
                p1 += 1
                p2 = 0

            else:
                p1 += 1
                p2 += 1

        if p2 == required_duration:
            return p1 - required_duration

        else:
            return -1

    def _occupy(self, state_matrix, state_matrix_capacity, required_resource, required_duration, start_time):
        """
        根据资源需求，将空闲资源占用

        :param state_matrix:
        :param state_matrix_capacity:
        :param required_resource:
        :param required_duration:
        :param start_time:
        :return:
        """

        for i in range(start_time, start_time + required_duration):
            for j in range(required_resource):
                state_matrix[i, len(state_matrix[i]) - state_matrix_capacity[i] + j] = 0
            state_matrix_capacity[i] -= required_resource

    def _expand(self, matrix, bg_shape=None):
        """
        将 state matrix 扩展为指定的 background shape
        :param matrix:
        :param bg_shape:
        :return:
        """

        if bg_shape is not None and bg_shape[0] >= matrix.shape[0] and bg_shape[1] >= matrix.shape[1]:
            temp = matrix
            if bg_shape[0] > matrix.shape[0]:
                temp = np.concatenate(
                    (temp, np.full(
                        (bg_shape[0] - matrix.shape[0]), 255,
                        dtype=np.uint8)), axis=1)
            if bg_shape[1] > matrix.shape[1]:
                temp = np.concatenate(
                    (temp, np.full(
                        (bg_shape[0], bg_shape[1] - matrix.shape[1]), 255,
                        dtype=np.uint8)), axis=1)
            return temp

        else:
            return matrix

    def __repr__(self):
        return 'Node(state_matrices={0}, label={1})'.format(self.state_matrices, self.label)
