"""算法的文本结果显示"""
import pprint
import sys
from collections import namedtuple

from models.utils.parameters import args


class SchedulingResult:
    """打印给定 job_num 中 task 的调度结果"""

    def __init__(self, cpu_earliest_finish_time_all, task_deployment_all, cpu_task_mapping_list_all,
                 task_start_time_all, job_num):
        self.job_num = job_num
        self.cpu_task_mapping_list_all = cpu_task_mapping_list_all
        self.task_deployment_all = task_deployment_all
        self.cpu_earliest_finish_time_all = cpu_earliest_finish_time_all
        self.task_start_time_all = task_start_time_all
        self.Event = namedtuple('Event', 'start end')

    def print(self):
        """
        打印给定 job 的 task 调度结果
        """
        task_deployment = self.task_deployment_all[self.job_num]
        cpu_earliest_finish_time = self.cpu_earliest_finish_time_all[self.job_num]
        cpu_task_mapping_list = self.cpu_task_mapping_list_all[self.job_num]
        task_start_time = self.task_start_time_all[self.job_num]

        schedules = [[] for _ in range(args.n_nodes)]
        for task_num in cpu_task_mapping_list:
            cpu_selected = int(task_deployment[task_num - 1])
            pair = {
                'task_num=' + str(task_num): self.Event(start=task_start_time[task_num - 1],
                                                        end=cpu_earliest_finish_time[task_num - 1][cpu_selected])
            }
            schedules[cpu_selected].append(pair)
        schedules_dict = {}
        for n in range(args.n_nodes):
            schedules_dict['cpu ' + str(n + 1)] = schedules[n]
        print('\nThe finish time of each task on the selected cpu for job #%d:' % self.job_num)
        pprint.pprint(schedules_dict)


class ProgressBar:
    """进度条"""

    def __init__(self, width=50):
        self.last = -1
        self.width = width

    def update(self, percent):
        assert 0 <= percent <= 100
        if self.last == percent:
            return
        self.last = int(percent)
        pos = int(self.width * (percent / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(percent), '#' * pos + '.' * (self.width - pos)))
        sys.stdout.flush()
        if percent == 100:
            print('')
