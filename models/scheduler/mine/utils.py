import json
import os
import random

from task import Task


def _generate_tasks():
    """
    根据 conf/task.pattern.conf.json 文件生成 task.csv"
    :return:
    """
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


def _load_tasks():
    """
    从 __cache__/tasks.csv 加载任务

    :return:
    """
    _generate_tasks()
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
