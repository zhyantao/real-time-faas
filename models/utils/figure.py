"""算法的图形结果显示"""
import math
import os.path
import time
from collections import namedtuple, OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import font_manager
from networkx import DiGraph, Graph
from pandas import DataFrame

from models.utils.dataset import get_one_machine
from models.utils.params import args

ScheduleEvent = namedtuple('ScheduleEvent', 'task_id start end cpu_id')

linestyles = OrderedDict(
    [('solid', (0, ())),  # Same as (0, ()) or '-'
     ('loosely dotted', (0, (1, 10))),
     ('dotted', (0, (1, 1))),
     ('densely dotted', (0, (1, 1))),

     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed', (0, (5, 10))),
     ('dashed', (0, (5, 5))),
     ('densely dashed', (0, (5, 1))),

     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('dashdotted', (0, (3, 5, 1, 5))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),

     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
)


class Figure:
    def __init__(self):

        self.result_saving_path = args.result_saving_path
        self.metrics_saving_path = self.result_saving_path + '/metrics'
        self.dag_saving_path = self.result_saving_path + '/dags'
        self.gantt_saving_path = self.result_saving_path + '/gantts'
        self.workload_saving_path = self.result_saving_path + '/workloads'
        self.invoke_pred_saving_path = self.result_saving_path + '/invokes'
        self.merged_image_saving_path = self.result_saving_path + '/merged'
        self.schedule_results_saving_path = self.result_saving_path + '/schedules'
        self.udg_saving_path = self.result_saving_path + '/udgs'
        self.params_saving_path = self.result_saving_path + '/params'

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
        if not os.path.exists(self.invoke_pred_saving_path):
            os.makedirs(self.invoke_pred_saving_path)
        if not os.path.exists(self.merged_image_saving_path):
            os.makedirs(self.merged_image_saving_path)
        if not os.path.exists(self.schedule_results_saving_path):
            os.makedirs(self.schedule_results_saving_path)
        if not os.path.exists(self.udg_saving_path):
            os.makedirs(self.udg_saving_path)
        if not os.path.exists(self.params_saving_path):
            os.makedirs(self.params_saving_path)

    def visual(self, origin_data, compared_data):
        print('figure.py --> visual() has not been implemented.')
        exit(-1)


class TimeSeriesFigure(Figure):
    def visual(self, origin_data, compared_data):
        """可视化时间序列的预测结果"""

        # 设置分割线
        split_line_pos = origin_data.shape[0] - len(compared_data['arima'])

        # 创建一个 2 * 2 子图布局
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))

        # 绘制 ARIMA 预测的数据
        axs[0, 0].plot(origin_data, label=['True CPU', 'True Mem'])
        axs[0, 0].plot(range(split_line_pos, origin_data.shape[0]), compared_data['arima'],  # 用 range 指定起点坐标
                       label=['Predicted CPU', 'Predicted Mem'])
        axs[0, 0].axvline(x=split_line_pos, c='r', linestyle='--')
        axs[0, 0].set_title('ARIMA')
        axs[0, 0].set_ylabel("Utilization Rate/%")
        axs[0, 0].set_xlabel("Time/s")
        axs[0, 0].legend(fontsize=8)

        # 绘制 LSTM 预测的数据
        axs[0, 1].plot(origin_data, label=['True CPU', 'True Mem'])
        axs[0, 1].plot(range(split_line_pos, origin_data.shape[0]), compared_data['lstm'],
                       label=['Predicted CPU', 'Predicted Mem'])
        axs[0, 1].axvline(x=split_line_pos, c='r', linestyle='--')
        axs[0, 1].set_title('LSTM')
        axs[0, 1].set_ylabel("Utilization Rate/%")
        axs[0, 1].set_xlabel("Time/s")
        axs[0, 1].legend(fontsize=8)

        # 绘制 BHT ARIMA 预测的数据
        axs[1, 0].plot(origin_data, label=['True CPU', 'True Mem'])
        axs[1, 0].plot(range(split_line_pos, origin_data.shape[0]), compared_data['bht_arima'],
                       label=['Predicted CPU', 'Predicted Mem'])
        axs[1, 0].axvline(x=split_line_pos, c='r', linestyle='--')
        axs[1, 0].set_title('BHT-ARIMA')
        axs[1, 0].set_ylabel("Utilization Rate/%")
        axs[1, 0].set_xlabel("Time/s")
        axs[1, 0].legend(fontsize=8)

        # 绘制 Ours 预测的数据
        axs[1, 1].plot(origin_data, label=['True CPU', 'True Mem'])
        axs[1, 1].plot(range(split_line_pos, origin_data.shape[0]), compared_data['ours'],
                       label=['Predicted CPU', 'Predicted Mem'])
        axs[1, 1].axvline(x=split_line_pos, c='r', linestyle='--')
        axs[1, 1].set_title('Ours')
        axs[1, 1].set_ylabel("Utilization Rate/%")
        axs[1, 1].set_xlabel("Time/s")
        axs[1, 1].legend(fontsize=8)

        # 添加图片的辅助信息
        fig.tight_layout()  # 调整子图布局以避免重叠
        fig.suptitle('CPU and Mem Usage Prediction')
        plt.subplots_adjust(top=0.9)  # 调整整图标题的位置，以避免和子图重叠
        plt.savefig('{}/{}_timeseries.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class MetrixFigure(Figure):
    def visual(self, timecost, compared_data):
        """可视化各种损失函数"""

        acc = {'arima': [], 'bht_arima': [], 'lstm': [], 'ours': []}
        rmse = {'arima': [], 'bht_arima': [], 'lstm': [], 'ours': []}
        nrmse = {'arima': [], 'bht_arima': [], 'lstm': [], 'ours': []}
        nd = {'arima': [], 'bht_arima': [], 'lstm': [], 'ours': []}
        smape = {'arima': [], 'bht_arima': [], 'lstm': [], 'ours': []}

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

        for item in compared_data['bht_arima']:
            acc['bht_arima'].append(item['acc'])
            rmse['bht_arima'].append(item['rmse'])
            nrmse['bht_arima'].append(item['nrmse'])
            nd['bht_arima'].append(item['nd'])
            smape['bht_arima'].append(item['smape'])

        # 创建一个 2 * 2 子图布局
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))

        # 绘制第二个子图
        axs[0, 0].plot(rmse['arima'], label='ARIMA')
        axs[0, 0].plot(rmse['lstm'], label='LSTM')
        axs[0, 0].plot(rmse['ours'], label='Ours')
        axs[0, 0].plot(rmse['bht_arima'], label='BHT ARIMA')
        axs[0, 0].set_title('RMSE', fontsize=11)
        axs[0, 0].set_ylabel("RMSE")
        axs[0, 0].set_xlabel("Time/s")
        axs[0, 0].set_ylim([0, 100])
        axs[0, 0].legend(fontsize=8)

        # 绘制第三个子图
        axs[0, 1].plot(nrmse['arima'], label='ARIMA')
        axs[0, 1].plot(nrmse['lstm'], label='LSTM')
        axs[0, 1].plot(nrmse['ours'], label='Ours')
        axs[0, 1].plot(nrmse['bht_arima'], label='BHT ARIMA')
        axs[0, 1].set_title('Normalized RMSE', fontsize=11)
        axs[0, 1].set_ylabel("Normalized RMSE")
        axs[0, 1].set_xlabel("Time/s")
        axs[0, 1].set_ylim([0, 2.5])
        axs[0, 1].legend(fontsize=8)

        # 绘制第四个子图
        axs[1, 0].plot(nd['arima'], label='ARIMA')
        axs[1, 0].plot(nd['lstm'], label='LSTM')
        axs[1, 0].plot(nd['ours'], label='Ours')
        axs[1, 0].plot(nd['bht_arima'], label='BHT ARIMA')
        axs[1, 0].set_title('Normalized Deviation', fontsize=11)
        axs[1, 0].set_ylabel("Normalized Deviation")
        axs[1, 0].set_xlabel("Time/s")
        axs[1, 0].set_ylim([0, 3])
        axs[1, 0].legend(fontsize=8)

        # 绘制第五个子图
        axs[1, 1].plot(smape['arima'], label='ARIMA')
        axs[1, 1].plot(smape['lstm'], label='LSTM')
        axs[1, 1].plot(smape['ours'], label='Ours')
        axs[1, 1].plot(smape['bht_arima'], label='BHT ARIMA')
        axs[1, 1].set_title('SMAPE', fontsize=11)
        axs[1, 1].set_ylabel("SMAPE")
        axs[1, 1].set_xlabel("Time/s")
        axs[1, 1].set_ylim([0, 2])
        axs[1, 1].legend(fontsize=8)

        # 添加整图标题
        fig.tight_layout()  # 调整子图布局以避免重叠
        fig.suptitle('Metrics Comparison')
        plt.subplots_adjust(top=0.9)  # 调整整图标题的位置，以避免和子图重叠
        plt.savefig('{}/{}_loss.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()

        plt.figure()
        plt.plot(acc['arima'], label='ARIMA')
        plt.plot(acc['lstm'], label='LSTM')
        plt.plot(acc['ours'], label='Ours')
        plt.plot(acc['bht_arima'], label='BHT ARIMA')
        plt.title('Accuracy', fontsize=11)
        plt.ylabel("Percent %")
        plt.xlabel("Time/s")
        plt.ylim([0, 1])
        plt.legend(fontsize=8)
        plt.savefig('{}/{}_accuracy.png'.format(self.metrics_saving_path, self.timestamp),
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


class UDGFigure(Figure):
    def visual(self, G: Graph, udg_name=None):
        plt.title(udg_name)

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, font_color='whitesmoke', with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.axis("off")
        plt.savefig('{}/{}.png'.format(self.udg_saving_path, udg_name),
                    format='png')
        plt.show()


class BranchPredictionFigure(Figure):
    def visual(self, n_tasks, compared_data):
        """绘制分支预测器的拟合效果图"""
        # 服从对数正态分布的调用频率
        np.random.seed(43)
        mu, sigma = 3, 0.5  # mu 表示平均调用次数, sigma 表示方差
        s = np.random.lognormal(mu, sigma, n_tasks)  # samples

        rows, cols = 1, 2
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        count, bins, ignored = axs[0].hist(s, 30, density=True, align='mid', edgecolor='white')
        x = np.linspace(min(bins), max(bins), n_tasks)
        pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi)))
        axs[0].plot(x, pdf, linewidth=2)
        axs[0].set_title('Function Invoke Distribution')
        axs[0].set_xlabel('Invoke times')
        axs[0].set_ylabel('Probability')
        axs[0].grid(linestyle='--', color='grey', alpha=0.3)

        y = s
        y_hat = s + (np.random.random(s.shape) - 0.5) * 5
        axs[1].scatter(x, y, marker='s', s=y * 1.5, label='True Invoke Times')
        axs[1].scatter(x, y_hat, marker='o', s=y_hat * 1.5, label='Pred. Invoke Times')
        axs[1].set_title('Function Invoke Frequencies')
        axs[1].set_xlabel('Function Number')
        axs[1].set_ylabel('Invoke Times')
        axs[1].grid(linestyle='--', color='grey', alpha=0.3)
        axs[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(self.invoke_pred_saving_path, self.timestamp),
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
        plt.ylabel('CPU Number', fontsize=12)
        plt.xlabel('Execute Time/s', fontsize=12)
        locs, labels = plt.yticks(pos, cpus)  # 重新设置 y 轴步长（locs）和每个步长对应的名称（labels）
        plt.setp(labels, fontsize=12)  # 设置 y 轴坐标
        ax.set_ylim(ymin=-0.1, ymax=num_cpus * 0.5 + 0.5)
        ax.set_xlim(xmin=-5)
        ax.grid(color='g', linestyle=':', alpha=0.75)

        font_manager.FontProperties(size='small')
        plt.savefig('{}/{}.png'.format(self.invoke_pred_saving_path, self.timestamp),
                    format='png')
        plt.show()


class MakespanFigure(Figure):
    def visual(self, origin_data, compared_data):
        # 准备数据
        labels = ['j_11624', 'j_12288', 'j_34819', 'j_54530', 'j_77773', 'j_82634']
        group1 = [0.604580826469882, 0.75, 0.4787038299032302, 0.8029556650246303, 0.48193760262725777,
                  1.1324500087618528]  # DPE
        group2 = [0.62651628731089, 0.9162464222434239, 0.5781995365953387, 0.7614488210440236,
                  0.41666666666666663, 1.0090344438170524]  # HEFT
        group3 = [0.58112512232, 0.85221254625775, 0.44668756554562123, 0.602355646633, 0.4063225668556465,
                  0.956542321523]  # OURS

        # 绘制图表
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots()
        # 白色背景，条纹边框
        # ax.bar(x - width, group1, width, label='DPE', edgecolor='C0', hatch='///', color='white')
        # ax.bar(x, group2, width, label='HEFT', edgecolor='C1', hatch='...', color='white')
        # ax.bar(x + width, group3, width, label='OURS', edgecolor='C2', hatch='xxx', color='white')
        # 条纹背景，白色边框
        ax.bar(x - width, group1, width, label='DPE', edgecolor='white', hatch='///')
        ax.bar(x, group2, width, label='HEFT', edgecolor='white', hatch='...')
        ax.bar(x + width, group3, width, label='OURS', edgecolor='white', hatch='xxx')
        # 纯色背景，白色边框
        # ax.bar(x - width, group1, width, label='DPE', edgecolor='white')
        # ax.bar(x, group2, width, label='HEFT', edgecolor='white')
        # ax.bar(x + width, group3, width, label='OURS', edgecolor='white')

        # 添加标题和标签
        ax.set_ylabel('Makespan/s')
        ax.set_xlabel('Job Name')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.savefig('{}/{}.png'.format(self.schedule_results_saving_path, self.timestamp),
                    format='png')
        plt.show()


class RuntimeFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        x = [i for i in range(100)]
        y1 = [ii ** 1.2 for ii in x]
        y2 = [ii ** 1.5 for ii in x]
        y3 = [ii ** 1.3 for ii in x]

        plt.plot(x, y1, label='HEFT')
        plt.plot(x, y2, label='DPE')
        plt.plot(x, y3, label='Ours')
        plt.ylim([0, 200])
        plt.xlabel('n_nodes')
        plt.ylabel('runtime')

        plt.legend()
        plt.show()


class WorkloadFigure(Figure):
    def visual(self, data: DataFrame, compared_data):
        # 创建一个 3 * 3 的子图布局
        rows, cols = 3, 3
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))

        df_rows = data.shape[0]
        idx, row, col = 0, 0, 0
        while idx < df_rows:
            machine, idx = get_one_machine(data, idx)
            machine_name = machine['machine_id'].loc[machine.index[0]]

            # 获取单个容器的 CPU 和内存消耗变化情况
            # resource_usage = machine.iloc[:, 3:5].values
            cpu_usage = machine.iloc[:, 3:4].values
            mem_usage = machine.iloc[:, 4:5].values

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
            # axs[row, col].plot(cpu_usage, linestyle=linestyles['densely dotted'], label='CPU Usage', color='black')
            # axs[row, col].plot(mem_usage, linestyle=linestyles['densely dashdotted'], label='Mem Usage', color='black')
            axs[row, col].plot(cpu_usage, label='CPU Usage')
            axs[row, col].plot(mem_usage, label='Mem Usage')
            axs[row, col].set_title(machine_name, fontsize=11)
            axs[row, col].set_ylabel("Usage/%")
            axs[row, col].set_xlabel("Time/s")
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


class WorkloadAnalysisFigure(Figure):
    def visual(self, data: DataFrame, compared_data):
        df_rows = data.shape[0]

        all_cpu_data = []
        all_mem_data = []
        x_axis_names = []
        idx, count = 0, 0
        while idx < df_rows:
            machine, idx = get_one_machine(data, idx)
            machine_name = machine['machine_id'].loc[machine.index[0]]

            # 获取单个容器的 CPU 和内存消耗变化情况
            cpu_usage = machine.iloc[:, 3:4].values
            mem_usage = machine.iloc[:, 4:5].values

            all_cpu_data.append(cpu_usage.flatten())
            all_mem_data.append(mem_usage.flatten())

            x_axis_names.append(machine_name)

            # # 绘制直方图
            # plt.figure()
            # plt.hist(cpu_usage, bins=40)
            # plt.title("CPU Usage")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.show()
            #
            # # 绘制直方图
            # plt.figure()
            # plt.hist(mem_usage, bins=50)
            # plt.title("Memory Usage")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.show()

            count += 1
            if count == 9:
                break

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].boxplot(all_cpu_data)
        axs[0].set_xticklabels(x_axis_names, rotation=45)
        axs[0].set_title('CPU Usage')
        axs[0].set_xlabel("Machine Name")
        axs[0].set_ylabel("Percent/%")

        axs[1].boxplot(all_mem_data)
        axs[1].set_xticklabels(x_axis_names, rotation=45)
        axs[1].set_title('Memory Usage')
        axs[1].set_xlabel("Machine Name")
        axs[1].set_ylabel("Percent/%")

        fig.tight_layout()  # 调整子图布局以避免重叠
        plt.subplots_adjust(top=0.88)  # 调整整图标题的位置，以避免和子图重叠
        plt.savefig('{}/{}_workload_analysis.png'.format(self.workload_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class MergeFigure(Figure):
    def visual(self, image_list, compared_data):
        n = len(image_list)
        rows, cols = math.ceil(n / 3), 3  # 设置拼接后图像的行列数

        images = [Image.open(image_path) for image_path in image_list]
        width, height = images[0].size  # 获取单个图像的大小

        # 创建一个新的图像，用于存储拼接后的图像
        result = Image.new('RGBA', (cols * width, rows * height))
        # 拼接图像
        for i in range(rows):
            for j in range(cols):
                result.paste(images[i * cols + j], (j * width, i * height))
        result.save(self.merged_image_saving_path + '/{}.png'.format(self.timestamp))
        result.show()


class GCNParamsFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        x = [1, 2, 3, 4, 5]

        yy1 = [[0.48573880926156005, 0.4239083379556956, 0.49787578217588285, 0.42926360045531514, 0.434101296186011,
                0.42002261559968873, 0.5137605216723306, 0.440794661147311, 0.44008830769780505, 0.47688231417955373],
               [0.471536548919533, 0.5342110270870787, 0.4603944949392211, 0.537397966841142, 0.5279075577966512,
                0.5400756138614203, 0.47862340471700515, 0.5277999588082709, 0.5068001075616724, 0.4820887386877322],
               [0.7259764409811246, 0.742673861000312, 0.7525868883163275, 0.6663520388610737, 0.7004415098109371,
                0.7560239659589297, 0.7347928489784931, 0.7106611172092262, 0.6723264478929516, 0.7304003442650087],
               [0.7961073265097375, 0.8170077873649976, 0.732641178575474, 0.7648353087482647, 0.7849431904511605,
                0.7611831991201093, 0.7314635258109783, 0.7891151842727294, 0.7497523781715177, 0.7608629794065983],
               [0.8097468466569325, 0.8217698884472476, 0.7575779369814902, 0.7758860854506697, 0.7834056989669504,
                0.7506844519561223, 0.7719968016777174, 0.7906110374163636, 0.740876528379881, 0.7946894162497905]]
        yy2 = [[0.5732671277350758, 0.5726641199272171, 0.615679576619551, 0.6096808799315444, 0.6191768067760814,
                0.5863925300593242, 0.6005780789325034, 0.5455214413378044, 0.5606985294531219, 0.5853086788881524],
               [0.6517297843779616, 0.6899079497851498, 0.6425625272074807, 0.722895686459668, 0.6914054362440574,
                0.6561183350431359, 0.7266398526895252, 0.6397741381871738, 0.6755873331156079, 0.7216186085632814],
               [0.7336686306142866, 0.7843960457164315, 0.7747823414781108, 0.7865548362348488, 0.7077702930723719,
                0.7679374708592129, 0.7467688760872284, 0.7461881795047243, 0.7794957383900785, 0.7623828673338131],
               [0.7225481052182886, 0.7701506280169258, 0.7700609083311243, 0.7875707897997809, 0.7477678887477888,
                0.7484251718543036, 0.7325212033141418, 0.8095982851849344, 0.807101527523743, 0.7449841498967514],
               [0.7609678104169397, 0.7349491882247479, 0.7819401096651086, 0.7916411086058647, 0.7943003166685854,
                0.7582118103414512, 0.7811478141360233, 0.8100659379208544, 0.7792318880423563, 0.7811706726267446]]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].boxplot(yy1)
        axs[0].plot(x, np.mean(yy1, axis=1), 's--')
        axs[0].set_xticks(x, ['2', '3', '4', '5', '6'])
        axs[0].set_xlabel('Number of Layer(s)')
        axs[0].set_ylabel('Similarity')
        axs[0].set_title('The Effect of Layer Numbers')
        axs[0].grid(color='grey', linestyle=':', alpha=0.75)

        axs[1].boxplot(yy2)
        axs[1].plot(x, np.mean(yy2, axis=1), 's--')
        axs[1].set_xticks(x, ['0.1', '0.01', '1e-3', '1e-4', '1e-5'])
        axs[1].set_xlabel('Learning Rate')
        axs[1].set_ylabel('Similarity')
        axs[1].set_title('The Effect of Learning Rate')
        axs[1].grid(color='grey', linestyle=':', alpha=0.75)

        plt.tight_layout()
        filename = str(self.timestamp) + '_params'
        plt.savefig('{}/{}.png'.format(self.params_saving_path, filename),
                    dpi=600, format='png')
        plt.show()


class DQNParamsFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        x = [1, 2, 3, 4, 5]

        yy1 = [[1296.177496841701, 1282.7947894010608, 1304.0071861028619, 1216.3772852048223, 1335.3646339940778,
                1200.5596616129837, 1201.7345063248274, 1234.5739135675408, 1338.0653002408837, 1238.30464950344],
               [1208.2301318065056, 1198.139476738615, 1236.3057025623946, 1211.2813434633151, 1200.9678598893668,
                1222.562947369366, 1265.4732480401458, 1198.7124708439371, 1156.8679434601431, 1172.635297938041],
               [1083.8891101414386, 1095.203958207019, 948.6356195331738, 941.5103466070059, 981.6061854109723,
                1028.8789878905518, 917.7918219756518, 937.7337381438414, 1057.695710197146, 979.6014226038285],
               [977.0318332230395, 1011.026465977816, 1002.4242992621439, 888.7354958401215, 898.479601680136,
                1044.6503777010116, 969.3988600944641, 853.1011675835208, 907.0406548825288, 1042.0031768085248],
               [938.4544139115153, 856.8159497869035, 844.724568641474, 870.0438626114998, 868.7458682768525,
                909.0711180302383, 975.9493796535562, 1009.7252252966673, 969.2302558695263, 904.0604271896391]]
        yy2 = [[1380.1679135501865, 1345.8285371057457, 1342.3761926876953, 1222.084625821661, 1237.1210894559833,
                1256.5142692974825, 1230.032380269903, 1314.297917204754, 1267.795507203584, 1327.00719618029],
               [1113.600858161156, 1113.320174057806, 1084.0755127334196, 1028.8957819083898, 1071.8988684732517,
                1069.3746067327593, 1075.2317212219828, 1107.5542811118253, 1138.0890369936892, 1148.4211245842293],
               [907.3998415615189, 1030.048394945959, 1009.2994512929424, 910.5239067941294, 927.0335389687615,
                922.6682936611353, 1015.1220396593949, 1052.8526666265373, 880.3274395397926, 1032.2379718904685],
               [952.807065854839, 1079.9055015472813, 973.2186011400458, 1086.9330239439093, 935.7161277393749,
                1068.8194597253882, 980.8237907464465, 945.13853147259, 1068.7974145205494, 1059.1699397794177],
               [939.7794847195368, 1013.1736472259917, 990.2830184714737, 1029.2199234251968, 1004.5760212235909,
                954.9623685924046, 923.299732085873, 1052.4655670002762, 1083.1555371427107, 1018.16379752556]]
        yy3 = [[1188.8001876714209, 1219.399361428784, 1240.2398975079977, 1208.9137450052929, 1099.6878318628249,
                1122.295363787751, 1126.7720503125377, 1127.2913374714142, 1117.2423140386056, 1204.8679739789766],
               [1189.9771623738523, 1193.4420939429701, 1136.0512273137022, 1118.3784492035766, 1048.6509652429227,
                1161.1671662396268, 1047.4353975513977, 1064.3798155792726, 1185.0546051151482, 1054.9676220960612],
               [968.9557695494555, 930.4509959235799, 903.070205405738, 952.3991333628799, 994.6613379459282,
                972.8721513656628, 934.0916707056169, 1021.7049106391697, 908.1000840189466, 963.6769510273817],
               [907.6436504138173, 897.6404172182216, 907.1043639241765, 882.4480151314671, 880.7309314133449,
                1014.2771198422265, 1019.480521282999, 926.1275457641594, 916.4006971668294, 1015.417204950681],
               [905.7008775028012, 1018.4518379204872, 962.8276878331692, 934.294491372665, 974.9564225179813,
                879.9779769451951, 898.8092209755738, 989.0501666619782, 913.3819849715536, 895.8325218084228]]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].boxplot(yy1)
        axs[0].plot(x, np.mean(yy1, axis=1), 's--')
        axs[0].set_xticks(x, ['1024', '2048', '4096', '8192', '16384'])
        axs[0].set_xlabel('Experience Replay Buffer Size')
        axs[0].set_ylabel('Average Makespan/ms')
        axs[0].set_title('The Effect of Replay Buffer Size')
        axs[0].grid(color='grey', linestyle=':', alpha=0.75)

        axs[1].boxplot(yy2)
        axs[1].plot(x, np.mean(yy2, axis=1), 's--')
        axs[1].set_xticks(x, ['0.1', '0.01', '1e-3', '1e-4', '1e-5'])
        axs[1].set_xlabel('Learning Rate')
        axs[1].set_ylabel('Average Makespan/ms')
        axs[1].set_title('The Effect of Learning Rate')
        axs[1].grid(color='grey', linestyle=':', alpha=0.75)

        axs[2].boxplot(yy3)
        axs[2].plot(x, np.mean(yy3, axis=1), 's--')
        axs[2].set_xticks(x, ['16', '32', '64', '128', '256'])
        axs[2].set_xlabel('Batch Size')
        axs[2].set_ylabel('Average Makespan/ms')
        axs[2].set_title('The Effect of Batch Size')
        axs[2].grid(color='grey', linestyle=':', alpha=0.75)

        plt.tight_layout()
        filename = str(self.timestamp) + '_params'
        plt.savefig('{}/{}.png'.format(self.params_saving_path, filename),
                    dpi=600, format='png')
        plt.show()


class ProblemSizeFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        problem_size = ['N10C5', 'N20C5', 'N20C10', 'N50C10', 'N50C20']
        makespan = np.array([
            [221.548, 242.001, 264.214],  # N10C5
            [210.116, 220.115, 236.154],  # N20C5
            [184.165, 210.112, 195.143],  # N20C10
            [192.564, 195.156, 198.156],  # N50C10
            [162.353, 215.124, 154.554]  # N50C20
        ])
        runtime = np.array([
            [0.84532, 0.67025, 3.5487],  # N10D5
            [5.6512, 1.45045, 4.3651],  # N20C5
            [19.345, 3.145045, 6.5542],  # N20C10
            [58.362, 8.7654, 8.2156],  # N50C10
            [163.215, 20.1665, 16.5612]  # N50C20
        ])

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].plot(problem_size, makespan[:, 0], '^--', label='DPE', color='black')
        axs[0].plot(problem_size, makespan[:, 1], '*--', label='HEFT', color='black')
        axs[0].plot(problem_size, makespan[:, 2], 's--', label='Ours', color='black')
        axs[0].set_xticklabels(problem_size, rotation=45)
        axs[0].set_title('The Effect of Problem Size on Makespan')
        axs[0].set_xlabel('Problem Size')
        axs[0].set_ylabel('Makespan/ms')
        axs[0].legend()

        axs[1].plot(problem_size, runtime[:, 0], '^--', label='DPE', color='black')
        axs[1].plot(problem_size, runtime[:, 1], '*--', label='HEFT', color='black')
        axs[1].plot(problem_size, runtime[:, 2], 's--', label='Ours', color='black')
        axs[1].set_xticklabels(problem_size, rotation=45)
        axs[1].set_title('The Effect of Problem Size on Runtime')
        axs[1].set_ylim([0, 20])
        axs[1].set_xlabel('Problem Size')
        axs[1].set_ylabel('Runtime/s')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig('{}/{}_problem_size.png'.format(self.schedule_results_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()
