"""论文插图"""

import os.path
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager
from networkx import DiGraph
from pandas import DataFrame

from models.utils.dataset import get_one_machine
from models.utils.params import args

ScheduleEvent = namedtuple('ScheduleEvent', 'task_id start end cpu_id')


class Figure:
    def __init__(self):

        self.result_saving_path = args.result_saving_path
        self.metrics_saving_path = self.result_saving_path + '/metrics'
        self.dag_saving_path = self.result_saving_path + '/dags'
        self.gantt_saving_path = self.result_saving_path + '/gantts'
        self.workload_saving_path = self.result_saving_path + '/workloads'

        self.timestamp = time.time()  # 用于区分不同时刻产生的结果文件

        # 检查是否存在保存实验结果的路径
        if not os.path.exists(self.result_saving_path):
            os.makedirs(self.result_saving_path)
        if not os.path.exists(self.metrics_saving_path):
            os.makedirs(self.metrics_saving_path)
        if not os.path.exists(self.dag_saving_path):
            os.makedirs(self.dag_saving_path)
        if not os.path.exists(self.gantt_saving_path):
            os.makedirs(self.gantt_saving_path)
        if not os.path.exists(self.workload_saving_path):
            os.makedirs(self.workload_saving_path)

    def visual(self, origin_data, compared_data):
        print('figure.py --> visual() has not been implemented.')
        exit(-1)


class TimeSeriesFigure(Figure):
    def visual(self, origin_data, compared_data):
        """可视化时间序列的预测结果"""

        # 设置分割线
        split_line_pos = origin_data.shape[0] - len(compared_data['arima'])
        plt.axvline(x=split_line_pos, c='r', linestyle='--')

        # 绘制真实数据
        plt.plot(origin_data, label=['Real CPU util.', 'Real Mem util.'])

        # 绘制 ARIMA 预测的数据（可选）
        plt.plot(range(split_line_pos, origin_data.shape[0]), compared_data['arima'],  # 用 range 指定起点坐标
                 label=['ARIMA pred. CPU util.', 'ARIMA pred. Mem util.'])

        # 绘制 LSTM 预测的数据（可选）
        plt.plot(range(split_line_pos, origin_data.shape[0]), compared_data['lstm'],
                 label=['LSTM pred. CPU util.', 'LSTM pred. Mem util.'])

        # 绘制 Ours 预测的数据（可选）
        plt.plot(range(split_line_pos, origin_data.shape[0]), compared_data['ours'],
                 label=['Ours pred. CPU util.', 'Ours pred. Mem util.'])

        # 添加图片的辅助信息
        plt.suptitle('CPU and Mem Usage Prediction')
        plt.ylabel("Utilization Rate (%)")
        plt.xlabel("Job Number (#)")
        plt.legend(fontsize=8)
        plt.savefig('{}/{}_timeseries.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class MetrixFigure(Figure):
    def visual(self, timecost, compared_data):
        """可视化各种损失函数"""

        acc = {'arima': [], 'lstm': [], 'ours': []}
        rmse = {'arima': [], 'lstm': [], 'ours': []}
        nrmse = {'arima': [], 'lstm': [], 'ours': []}
        nd = {'arima': [], 'lstm': [], 'ours': []}
        smape = {'arima': [], 'lstm': [], 'ours': []}

        for item in compared_data['arima']:
            acc['arima'].append(item['acc'])
            rmse['arima'].append(item['rmse'])
            nrmse['arima'].append(item['nrmse'])
            nd['arima'].append(item['nd'])
            smape['arima'].append(item['smape'])

        for item in compared_data['lstm']:
            acc['lstm'].append(item['acc'])
            rmse['lstm'].append(item['rmse'])
            nrmse['lstm'].append(item['nrmse'])
            nd['lstm'].append(item['nd'])
            smape['lstm'].append(item['smape'])

        for item in compared_data['ours']:
            acc['ours'].append(item['acc'])
            rmse['ours'].append(item['rmse'])
            nrmse['ours'].append(item['nrmse'])
            nd['ours'].append(item['nd'])
            smape['ours'].append(item['smape'])

        # 创建一个 2 * 3 子图布局
        rows, cols = 2, 3
        fig, axs = plt.subplots(rows, cols, figsize=(10, 5))

        # 绘制第一个子图
        axs[0, 0].plot(acc['arima'], label='ARIMA')
        axs[0, 0].plot(acc['lstm'], label='LSTM')
        axs[0, 0].plot(acc['ours'], label='Ours')
        axs[0, 0].set_title('Accuracy', fontsize=11)
        axs[0, 0].set_ylabel("Percent (%)")
        axs[0, 0].set_xlabel("Task Number (#)")
        axs[0, 0].legend(fontsize=8)

        # 绘制第二个子图
        axs[0, 1].plot(rmse['arima'], label='ARIMA')
        axs[0, 1].plot(rmse['lstm'], label='LSTM')
        axs[0, 1].plot(rmse['ours'], label='Ours')
        axs[0, 1].set_title('RMSE', fontsize=11)
        axs[0, 1].set_ylabel("RMSE")
        axs[0, 1].set_xlabel("Task Number (#)")
        axs[0, 1].legend(fontsize=8)

        # 绘制第三个子图
        axs[0, 2].plot(nrmse['arima'], label='ARIMA')
        axs[0, 2].plot(nrmse['lstm'], label='LSTM')
        axs[0, 2].plot(nrmse['ours'], label='Ours')
        axs[0, 2].set_title('Normalized RMSE', fontsize=11)
        axs[0, 2].set_ylabel("Normalized RMSE")
        axs[0, 2].set_xlabel("Task Number (#)")
        axs[0, 2].legend(fontsize=8)

        # 绘制第四个子图
        axs[1, 0].plot(nd['arima'], label='ARIMA')
        axs[1, 0].plot(nd['lstm'], label='LSTM')
        axs[1, 0].plot(nd['ours'], label='Ours')
        axs[1, 0].set_title('Normalized Deviation', fontsize=11)
        axs[1, 0].set_ylabel("Normalized Deviation")
        axs[1, 0].set_xlabel("Task Number (#)")
        axs[1, 0].legend(fontsize=8)

        # 绘制第五个子图
        axs[1, 1].plot(smape['arima'], label='ARIMA')
        axs[1, 1].plot(smape['lstm'], label='LSTM')
        axs[1, 1].plot(smape['ours'], label='Ours')
        axs[1, 1].set_title('SMAPE', fontsize=11)
        axs[1, 1].set_ylabel("SMAPE")
        axs[1, 1].set_xlabel("Task Number (#)")
        axs[1, 1].legend(fontsize=8)

        # 绘制第六个子图
        axs[1, 2].plot(timecost['arima'], label='ARIMA')
        axs[1, 2].plot(timecost['lstm'], label='LSTM')
        axs[1, 2].plot(timecost['ours'], label='Ours')
        axs[1, 2].set_title('Algo. exec. Time Cost', fontsize=11)
        axs[1, 2].set_ylabel("Time Cost (seconds)")
        axs[1, 2].set_xlabel("Task Number (#)")
        axs[1, 2].legend(fontsize=8)

        # 添加整图标题
        fig.tight_layout()  # 调整子图布局以避免重叠
        fig.suptitle('Metrics Comparison')
        plt.subplots_adjust(top=0.88)  # 调整整图标题的位置，以避免和子图重叠
        plt.savefig('{}/{}_loss.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class DAGFigure(Figure):
    def visual(self, G: DiGraph, job_name=None):
        # G: networkx 中的 DiGraph 格式
        plt.title(job_name)  # 配置属性必须在 nx.draw 函数调用之前

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, font_color='whitesmoke', with_labels=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.axis("off")
        plt.savefig('{}/{}.png'.format(self.dag_saving_path, job_name),
                    format='png')
        plt.show()


class GanttFigure(Figure):
    def __init__(self):
        super().__init__()

    def visual(self, schedule_events: dict, compared_data):
        """
        给定一个 cpu-task 映射表，使用甘特图将其可视化。比如：
        {
            cpu0: [task1(start_time, end_time), task2(start_time, end_time)],
            cpu1: [task4(start_time, end_time), task5(start_time, end_time), task3(start_time, end_time],
            ...
        }

        :param compared_data:
        :param schedule_events: cpu-task 映射表
        :return: 无返回值
        """

        cpus = list(schedule_events.keys())
        num_cpus = len(cpus)

        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)
        # 绘制水平柱状图
        for idx, cpu in enumerate(cpus):
            for event in schedule_events[cpu]:
                ax.barh((idx * 0.5) + 0.5, event.end - event.start, left=event.start, height=0.2,
                        align='center', edgecolor='black', color='white', alpha=0.8)
                ax.text(0.5 * (event.start + event.end - len(str(event.task_id))), 0.5 * idx + 0.47, event.task_id,
                        color='blue', fontweight='normal', fontsize=12, alpha=0.8)

        # 设置图像属性
        pos = np.arange(0.5, num_cpus * 0.5 + 0.5, 0.5)
        print(pos)
        plt.ylabel('CPU Number (#)', fontsize=12)
        plt.xlabel('Time Cost (s)', fontsize=12)
        locs, labels = plt.yticks(pos, cpus)  # 重新设置 y 轴步长（locs）和每个步长对应的名称（labels）
        plt.setp(labels, fontsize=12)  # 设置 y 轴坐标
        ax.set_ylim(ymin=-0.1, ymax=num_cpus * 0.5 + 0.5)
        ax.set_xlim(xmin=-5)
        ax.grid(color='g', linestyle=':', alpha=0.75)

        font_manager.FontProperties(size='small')
        plt.savefig('{}/{}.png'.format(self.gantt_saving_path, self.timestamp),
                    format='png')
        plt.show()


class WorkloadFigure(Figure):
    def visual(self, data: DataFrame, compared_data):
        # 创建一个 2 * 3 子图布局
        rows, cols = 3, 3
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))

        df_rows = data.shape[0]
        idx, row, col = 0, 0, 0
        while idx < df_rows:
            machine, idx = get_one_machine(data, idx)
            machine_name = machine['machine_id'].loc[machine.index[0]]

            # 获取单个容器的 CPU 和内存消耗变化情况
            resource_usage = machine.iloc[:, 3:5].values

            # # 在单张图像中展示
            # plt.figure()
            # plt.plot(resource_usage, label=['CPU Usage', 'Mem Usage'])
            # plt.title(machine_name, fontsize=11)
            # plt.ylabel("Usage (%)")
            # plt.xlabel("Time (s)")
            # plt.legend(fontsize=8)
            # plt.savefig('{}/{}_workload.png'.format(self.workload_saving_path, machine_name),
            #             dpi=600, format='png')
            # plt.close()

            # 在单张图像中展示多个子图
            axs[row, col].plot(resource_usage, label=['CPU Usage', 'Mem Usage'])
            axs[row, col].set_title(machine_name, fontsize=11)
            axs[row, col].set_ylabel("Usage (%)")
            axs[row, col].set_xlabel("Time (s)")
            axs[row, col].set_xticks([0, 30, 60, 90, 120, 150])
            axs[row, col].set_ylim(-10, 110)
            axs[row, col].grid(True, alpha=0.5, linewidth=0.5, linestyle='--')  # 显示水平和垂直辅助线
            axs[row, col].legend(fontsize=8)

            col += 1
            if col == cols:
                col = 0
                row += 1
                if row == rows:
                    break

        # 添加整图标题
        fig.tight_layout()  # 调整子图布局以避免重叠
        fig.suptitle('Resource Usage')
        plt.subplots_adjust(top=0.92)  # 调整整图标题的位置，以避免和子图重叠
        plt.savefig('{}/{}_workload.png'.format(self.workload_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


if __name__ == '__main__':
    # # 生成数据
    # x1 = np.random.rand(50)
    # y1 = np.random.rand(50)
    # x2 = np.random.rand(50)
    # y2 = np.random.rand(50)
    #
    # # 绘制折线图
    # plt.figure()
    # plt.plot(x1, y1, '-o', label='Line 1')
    # plt.plot(x2, y2, '-o', label='Line 2')
    # plt.legend()
    # plt.show()
    #
    # # 绘制散点图
    # plt.figure()
    # plt.scatter(x1, y1, color='red', marker='o')
    # plt.scatter(x2, y2, color='blue', marker='s')
    # plt.title('Multiple Scatter Plots')
    # plt.xlabel('X Axis')
    # plt.ylabel('Y Axis')
    # plt.show()
    #
    # # 重建 DAG
    # df = pd.read_csv(args.selected_batch_task_path)
    # idx = 0
    # while idx < df.shape[0]:
    #     job, idx = get_one_job(df, idx)
    #     G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
    #     dag_figure = DAGFigure()
    #     dag_figure.visual(G, job_name)
    #
    # mappings = {
    #     0: [ScheduleEvent(task_id=1000, start=0, end=14.0, cpu_id=0),
    #         ScheduleEvent(task_id=13, start=14.0, end=27.0, cpu_id=0),
    #         ScheduleEvent(task_id=1, start=27.0, end=40.0, cpu_id=0),
    #         ScheduleEvent(task_id=12, start=40.0, end=51.0, cpu_id=0),
    #         ScheduleEvent(task_id=7, start=57.0, end=62.0, cpu_id=0),
    #         ScheduleEvent(task_id=15, start=62.0, end=75.0, cpu_id=0),
    #         ScheduleEvent(task_id=16, start=75.0, end=82.0, cpu_id=0),
    #         ScheduleEvent(task_id=17, start=86.0, end=91.0, cpu_id=0)],
    #     1: [ScheduleEvent(task_id=3, start=18.0, end=26.0, cpu_id=1),
    #         ScheduleEvent(task_id=5, start=26.0, end=42.0, cpu_id=1),
    #         ScheduleEvent(task_id=14, start=42.0, end=55.0, cpu_id=1),
    #         ScheduleEvent(task_id=8, start=56.0, end=68.0, cpu_id=1),
    #         ScheduleEvent(task_id=9, start=73.0, end=80.0, cpu_id=1),
    #         ScheduleEvent(task_id=19, start=102.0, end=109.0, cpu_id=1)],
    #     2: [ScheduleEvent(task_id=0, start=0, end=9.0, cpu_id=2),
    #         ScheduleEvent(task_id=2, start=9.0, end=28.0, cpu_id=2),
    #         ScheduleEvent(task_id=4, start=28.0, end=38.0, cpu_id=2),
    #         ScheduleEvent(task_id=6, start=38.0, end=49.0, cpu_id=2),
    #         ScheduleEvent(task_id=11, start=49.0, end=67.0, cpu_id=2),
    #         ScheduleEvent(task_id=18, start=68.0, end=88.0, cpu_id=2)]
    # }
    #
    # gantt_figure = GanttFigure()
    # gantt_figure.visual(mappings, None)

    df = pd.read_csv(args.selected_container_usage_path)
    workload_figure = WorkloadFigure()
    workload_figure.visual(df, None)
