import os

import numpy as np
from PIL import Image

from task import Task


class Environment(object):
    """状态空间表示"""

    def __init__(self, nodes, queue_size, backlog_size, task_generator):
        """
        :param nodes: 节点数量
        :param queue_size: 队列大小
        :param backlog_size: 积压任务列表的大小
        :param task_generator: 任务生成器
        """
        self.nodes = nodes
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.queue = []  # 工作队列
        self.backlog = []  # 积压队列
        self.timestep_counter = 0
        self._task_generator = task_generator
        self._task_generator_end = False

    def timestep(self):
        """
        前进到下一个时间

        :return:
        """
        self.timestep_counter += 1

        # 每个 node 前进
        for node in self.nodes:
            node.timestep()

        # 将 task 从 backlog 移动到 queue
        p_queue = len(self.queue)
        p_backlog = 0
        indices = []
        while p_queue < self.queue_size and p_backlog < len(self.backlog):
            self.queue.append(self.backlog[p_backlog])
            indices.append(p_backlog)
            p_queue += 1
            p_backlog += 1
        # 删除积压工作列表中已添加到 queue 中的 task
        for i in sorted(indices, reverse=True):
            del self.backlog[i]

        # 当有更多 task 来临时，添加到 backlog
        p_backlog = len(self.backlog)
        while p_backlog < self.backlog_size:
            new_task = next(self._task_generator, None)
            if new_task is None:
                self._task_generator_end = True
                break
            else:
                self.backlog.append(new_task)
                p_backlog += 1

    def terminated(self):
        """
        检查是否可以终止算法了
        :return:
        """
        for node in self.nodes:
            if node.utilization() > 0:
                return False  # 如果还有剩余资源，说明不能停止
        if self.queue or self.backlog or not self._task_generator_end:
            return False  # 如果还是有剩余未分配的节点，不能停止
        return True

    def reward(self):
        """
        计算奖励

        :return:
        """
        r = 0
        for node in self.nodes:
            if node.scheduled_tasks:
                r += 1 / sum([task[0].duration for task in node.scheduled_tasks])
        if self.queue:
            r += 1 / sum([task.duration for task in self.queue])
        if self.backlog:
            r += 1 / sum([task.duration for task in self.backlog])
        return -r

    def summary(self, bg_shape=None):
        """
        状态表示
        :param bg_shape: background shape
        :return:
        """
        if bg_shape is None:
            bg_col = max([max(node.resources) for node in self.nodes])
            bg_row = max([node.duration for node in self.nodes])
            bg_shape = (bg_row, bg_col)

        if len(self.nodes) > 0:
            dimension = self.nodes[0].dimension

            # node 节点的状态表示
            temp = self.nodes[0].summary(bg_shape)
            for i in range(1, len(self.nodes)):
                temp = np.concatenate((temp, self.nodes[i].summary(bg_shape)), axis=1)

            # 已经占用资源的状态表示
            for i in range(len(self.queue)):
                temp = np.concatenate((temp, self.queue[i].summary(bg_shape)), axis=1)

            # 空闲资源的状态表示
            empty_summary = Task([0] * dimension, 0, 'empty_task').summary(bg_shape)
            for i in range(len(self.queue), self.queue_size):
                temp = np.concatenate((temp, empty_summary), axis=1)

            # 积压队列的状态表示
            backlog_summary = Task([0], 0, 'empty_task').summary(bg_shape)
            p_backlog = 0
            p_row = 0
            p_col = 0
            while p_row < bg_shape[0] and p_col < bg_shape[1] and p_backlog < len(self.backlog):
                backlog_summary[p_row, p_col] = 0
                p_row += 1
                if p_row == bg_shape[0]:
                    p_row = 0
                    p_col += 1
                p_backlog += 1
            temp = np.concatenate((temp, backlog_summary), axis=1)

            return temp
        else:
            return None

    def plot(self, bg_shape=None):
        """
        根据状态表示画图
        :param bg_shape:
        :return:
        """
        if not os.path.exists('__cache__/state'):
            os.makedirs('__cache__/state')
        summary_matrix = self.summary(bg_shape)
        summary_plot = np.full((summary_matrix.shape[0], summary_matrix.shape[1]), 255, dtype=np.uint8)
        for row in range(summary_matrix.shape[0]):
            for col in range(summary_matrix.shape[1]):
                summary_plot[row, col] = summary_matrix[row, col]
        Image.fromarray(summary_plot).save('__cache__/state/environment_{0}.png'.format(self.timestep_counter))

    def __repr__(self):
        return 'Environment(timestep_counter={0}, nodes={1}, queue={2}, backlog={3})'.format(
            self.timestep_counter, self.nodes, self.queue, self.backlog)
