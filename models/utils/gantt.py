"""
使用 Matplotlib 展示甘特图
"""
from collections import namedtuple

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

ScheduleEvent = namedtuple('ScheduleEvent', 'id start end proc')


def show(cpu_task_mappings):
    """
    给定一个 cpu-task 映射表，使用甘特图将其可视化。比如：
    {
        cpu0: [task1(start_time, end_time), task2(start_time, end_time)],
        cpu1: [task4(start_time, end_time), task5(start_time, end_time), task3(start_time, end_time],
        ...
    }

    :param cpu_task_mappings: cpu-task 映射表
    :return: 无返回值
    """

    cpus = list(cpu_task_mappings.keys())
    cpu_nums = len(cpus)

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    # 绘制水平柱状图
    for idx, cpu in enumerate(cpus):
        for task in cpu_task_mappings[cpu]:
            ax.barh((idx * 0.5) + 0.5, task.end - task.start, left=task.start, height=0.2,
                    align='center', edgecolor='black', color='white', alpha=0.8)
            ax.text(0.5 * (task.start + task.end - len(str(task.id))), 0.5 * idx + 0.47, task.id,
                    color='blue', fontweight='normal', fontsize=12, alpha=0.8)

    # 设置图像属性
    pos = np.arange(0.5, cpu_nums * 0.5 + 0.5, 0.5)
    print(pos)
    plt.ylabel('CPUs', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    locs, labels = plt.yticks(pos, cpus)  # 重新设置 y 轴步长（locs）和每个步长对应的名称（labels）
    plt.setp(labels, fontsize=12)  # 设置 y 轴坐标
    ax.set_ylim(ymin=-0.1, ymax=cpu_nums * 0.5 + 0.5)
    ax.set_xlim(xmin=-5)
    ax.grid(color='g', linestyle=':', alpha=0.75)

    font_manager.FontProperties(size='small')
    plt.show()


if __name__ == "__main__":
    mapping = {
        0: [ScheduleEvent(id=1000, start=0, end=14.0, proc=0), ScheduleEvent(id=13, start=14.0, end=27.0, proc=0),
            ScheduleEvent(id=1, start=27.0, end=40.0, proc=0), ScheduleEvent(id=12, start=40.0, end=51.0, proc=0),
            ScheduleEvent(id=7, start=57.0, end=62.0, proc=0), ScheduleEvent(id=15, start=62.0, end=75.0, proc=0),
            ScheduleEvent(id=16, start=75.0, end=82.0, proc=0), ScheduleEvent(id=17, start=86.0, end=91.0, proc=0)],
        1: [ScheduleEvent(id=3, start=18.0, end=26.0, proc=1), ScheduleEvent(id=5, start=26.0, end=42.0, proc=1),
            ScheduleEvent(id=14, start=42.0, end=55.0, proc=1), ScheduleEvent(id=8, start=56.0, end=68.0, proc=1),
            ScheduleEvent(id=9, start=73.0, end=80.0, proc=1),
            ScheduleEvent(id=19, start=102.0, end=109.0, proc=1)],
        2: [ScheduleEvent(id=0, start=0, end=9.0, proc=2), ScheduleEvent(id=2, start=9.0, end=28.0, proc=2),
            ScheduleEvent(id=4, start=28.0, end=38.0, proc=2), ScheduleEvent(id=6, start=38.0, end=49.0, proc=2),
            ScheduleEvent(id=11, start=49.0, end=67.0, proc=2), ScheduleEvent(id=18, start=68.0, end=88.0, proc=2)]}
    show(mapping)
