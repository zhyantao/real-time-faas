import numpy as np


class Task(object):
    """Task 实体类"""

    def __init__(self, resources, duration, label):
        """
        :param resources: 资源需求的种类 [cpu, memory, data_size]
        :param duration: 作业的总运行时间
        :param label: 作业的标签（task_name)
        """
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)  # 特征的数量

    def summary(self, bg_shape=None):
        """
        :param bg_shape: 状态表征
        :return:
        """
        if bg_shape is None:
            bg_shape = (self.duration, max(self.resources))

        if self.dimension > 0:
            state_matrices = [np.full(bg_shape, 255, dtype=np.uint8) for _ in range(self.dimension)]

            for i in range(self.dimension):
                for row in range(self.duration):
                    for col in range(self.resources[i]):
                        state_matrices[i][row, col] = 0

            temp = state_matrices[0]
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, state_matrices[i]), axis=1)

            return temp

        else:
            return None

    def __repr__(self):
        return 'Task(resources={0}, duration={1}, label={2})'.format(self.resources, self.duration, self.label)
