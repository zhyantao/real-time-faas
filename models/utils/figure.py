"""
可视化工具
"""
import pprint
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import yaml

with open("E:/Workshop/real-time-faas/configs/parameter.yaml", 'r') as f:
    para = yaml.load(f, Loader=yaml.FullLoader)


class ProgressBar:
    """
    进度条
    """

    def __init__(self, width=50):
        self.last = -1
        self.width = width

    def update(self, current):
        assert 0 <= current <= 100
        if self.last == current:
            return
        self.last = int(current)
        pos = int(self.width * (current / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(current), '#' * pos + '.' * (self.width - pos)))
        sys.stdout.flush()
        if current == 100:
            print('')


class LineChart:
    """
    折线图
    """

    def __init__(self, x_data, y_data, title, label, x_label, y_label):
        self.x_data = x_data  # 一维数组
        self.y_data = y_data  # 一维数组
        self.title = title  # 图名
        self.label = label  # 图例名称
        self.x_label = x_label  # x 轴名称
        self.y_label = y_label

    def show(self):
        x = self.x_data
        y = self.y_data

        plt.plot(x, y, "g", label=self.label)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()  # 设置图例的放置位置

        # # 在图上标注数字
        # for xx, yy in zip(x, y):
        #     plt.text(xx, yy, str(yy), ha="center", va="bottom", fontsize=10)

        plt.show()


class SchedulingResult:
    """
    打印给定 job_num 中 task 的调度结果
    """
    Event = namedtuple('Event', 'start end')

    def __init__(self, cpu_earliest_finish_time_all, task_deployment_all, cpu_task_mapping_list_all,
                 task_start_time_all, job_num):
        self.job_num = job_num
        self.cpu_task_mapping_list_all = cpu_task_mapping_list_all
        self.task_deployment_all = task_deployment_all
        self.cpu_earliest_finish_time_all = cpu_earliest_finish_time_all
        self.task_start_time_all = task_start_time_all

    def print(self):
        """
        打印给定 job 的 task 调度结果
        """
        task_deployment = self.task_deployment_all[self.job_num]
        cpu_earliest_finish_time = self.cpu_earliest_finish_time_all[self.job_num]
        cpu_task_mapping_list = self.cpu_task_mapping_list_all[self.job_num]
        task_start_time = self.task_start_time_all[self.job_num]

        schedules = [[] for _ in range(para.get("cpu_nums"))]
        for task_num in cpu_task_mapping_list:
            cpu_selected = int(task_deployment[task_num - 1])
            pair = {'task_num=' + str(task_num): self.Event(start=task_start_time[task_num - 1],
                                                            end=cpu_earliest_finish_time[task_num - 1][cpu_selected])}
            schedules[cpu_selected].append(pair)
        schedules_dict = {}
        for n in range(para.get("cpu_nums")):
            schedules_dict['cpu ' + str(n + 1)] = schedules[n]
        print('\nThe finish time of each task on the selected cpu for job #%d:' % self.job_num)
        pprint.pprint(schedules_dict)
