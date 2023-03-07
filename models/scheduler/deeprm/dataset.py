import json
import random

from models.scheduler.deeprm.state import Environment
from models.scheduler.deeprm.node import Node
from models.scheduler.deeprm.agent import *
from models.scheduler.deeprm.task import Task


class Dataset(object):

    def __init__(self):
        pass

    def load(self, load_environment=True, load_scheduler=True):
        """
        从 conf/env.conf.json 中加载环境和调度器

        :param load_environment:
        :param load_scheduler:
        :return:
        """
        tasks = self._load_tasks()  # [ Task(resources=[2, 3, 3], duration=2, label=task1), ... ]
        task_generator = (t for t in tasks)

        with open('../../configs/env.conf.json', 'r') as fr:
            data = json.load(fr)

            # 获取 node 信息
            nodes = []
            label = 0
            for node_json in data['nodes']:
                label += 1
                nodes.append(Node(node_json['resource_capacity'], node_json['duration_capacity'], 'node' + str(label)))

            # 获取 environment 信息
            environment = None
            if load_environment:
                environment = Environment(nodes, data['queue_size'], data['backlog_size'], task_generator)
                environment.timestep()  # 前进到下一个时间步

            # 获取 scheduler 信息
            scheduler = None
            if load_scheduler:
                if 'CompactScheduler' == data['scheduler']:
                    scheduler = CompactScheduler(environment)
                elif 'SpreadScheduler' == data['scheduler']:
                    scheduler = SpreadScheduler(environment)
                else:
                    scheduler = DeepRMScheduler(environment, data['train'])

            return environment, scheduler

    def _load_tasks(self):
        """Load tasks from __cache__/tasks.csv"""
        self._generate_tasks()
        tasks = []
        with open('__cache__/tasks.csv', 'r') as fr:
            resource_indices = []
            duration_index = 0
            label_index = 0
            line = fr.readline()
            parts = line.strip().split(',')
            for i in range(len(parts)):
                if parts[i].strip().startswith('resource'):
                    resource_indices.append(i)
                if parts[i].strip() == 'duration':
                    duration_index = i
                if parts[i].strip() == 'label':
                    label_index = i
            line = fr.readline()
            while line:
                parts = line.strip().split(',')
                resources = []
                for index in resource_indices:
                    resources.append(int(parts[index]))
                tasks.append(Task(resources, int(parts[duration_index]), parts[label_index]))
                line = fr.readline()
        return tasks

    def _generate_tasks(self):
        """根据 conf/task.pattern.conf.json 文件生成 task.csv"""
        if not os.path.exists('__cache__'):
            os.makedirs('__cache__')
        if os.path.isfile('__cache__/tasks.csv'):
            return
        with open('../../../configs/task.pattern.conf.json', 'r') as fr, open('__cache__/tasks.csv', 'w') as fw:
            data = json.load(fr)
            if len(data) > 0:
                for i in range(len(data[0]['resource_range'])):
                    fw.write('resource' + str(i + 1) + ',')
                fw.write('duration,label' + '\n')
            label = 0
            for task_pattern in data:
                for i in range(task_pattern['batch_size']):
                    label += 1
                    resources = []
                    duration = str(random.randint(task_pattern['duration_range']['lowerLimit'],
                                                  task_pattern['duration_range']['upperLimit']))
                    for j in range(len(task_pattern['resource_range'])):
                        resources.append(str(random.randint(task_pattern['resource_range'][j]['lowerLimit'],
                                                            task_pattern['resource_range'][j]['upperLimit'])))
                    fw.write(','.join(resources) + ',' + duration + ',' + 'task' + str(label) + '\n')
