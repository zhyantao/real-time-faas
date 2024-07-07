"""算法的图形结果显示"""
import math
import os.path
import time
from collections import namedtuple, OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import font_manager, gridspec
from networkx import DiGraph, Graph
from pandas import DataFrame

from models.utils.tools import get_one_machine
from models.utils.parameters import args

# 设置图片中的中文字体、英文字体、公式字体
FILE_PATH = os.path.abspath(__file__)  # 获取当前文件所在路径
CURRENT_DIR = os.path.dirname(FILE_PATH)  # 获取当前文件所在目录
font_path = CURRENT_DIR + "/../../fonts/tnw+simsun.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
plt.rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
plt.rcParams['font.size'] = 10  # 设置字体大小 (五号字体)
plt.rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号
plt.rcParams['mathtext.fontset'] = 'cm'  # 设置公式字体

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

        now = time.time()
        timeArray = time.localtime(now)
        otherStyleTime = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
        self.timestamp = otherStyleTime  # 用于区分不同时刻产生的结果文件

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

        # # 创建一个包含 5 个子图的布局：第一行 3 张图，第二行 2 张图
        plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 6)
        # gs.update(wspace=5)

        # 绘制 ARIMA 预测的数据
        ax1 = plt.subplot(gs[0, :2])
        ax1.plot(origin_data, label=['True CPU', 'True Mem'])
        ax1.plot(range(split_line_pos, origin_data.shape[0]), compared_data['arima'],  # 用 range 指定起点坐标
                 label=['Predicted CPU', 'Predicted Mem'])
        ax1.axvline(x=split_line_pos, c='r', linestyle='--')
        ax1.set_title('(a) ARIMA', y=-0.25)
        ax1.set_ylabel("Utilization Rate/%")
        ax1.set_xlabel("Time/s")
        ax1.legend(fontsize=8)

        # 绘制 LSTM 预测的数据
        ax2 = plt.subplot(gs[0, 2:4])
        ax2.plot(origin_data, label=['True CPU', 'True Mem'])
        ax2.plot(range(split_line_pos, origin_data.shape[0]), compared_data['lstm'],
                 label=['Predicted CPU', 'Predicted Mem'])
        ax2.axvline(x=split_line_pos, c='r', linestyle='--')
        ax2.set_title('(b) LSTM', y=-0.25)
        ax2.set_ylabel("Utilization Rate/%")
        ax2.set_xlabel("Time/s")
        ax2.legend(fontsize=8)

        # 绘制 BHT ARIMA 预测的数据
        ax3 = plt.subplot(gs[0, 4:6])
        ax3.plot(origin_data, label=['True CPU', 'True Mem'])
        ax3.plot(range(split_line_pos, origin_data.shape[0]), compared_data['bht_arima'],
                 label=['Predicted CPU', 'Predicted Mem'])
        ax3.axvline(x=split_line_pos, c='r', linestyle='--')
        ax3.set_title('(c) BHT-ARIMA', y=-0.25)
        ax3.set_ylabel("Utilization Rate/%")
        ax3.set_xlabel("Time/s")
        ax3.legend(fontsize=8)

        # 绘制 DLinear 预测的数据
        ax4 = plt.subplot(gs[1, 1:3])
        ax4.plot(origin_data, label=['True CPU', 'True Mem'])
        start_x_pos = origin_data.shape[0] - len(compared_data['dlinear'])
        ax4.plot(range(start_x_pos, origin_data.shape[0]), compared_data['dlinear'],
                 label=['Predicted CPU', 'Predicted Mem'])
        ax4.axvline(x=start_x_pos, c='r', linestyle='--')
        ax4.set_title('(d) DLinear', y=-0.25)
        ax4.set_ylabel("Utilization Rate/%")
        ax4.set_xlabel("Time/s")
        ax4.legend(fontsize=8)

        # 绘制 Ours 预测的数据
        ax5 = plt.subplot(gs[1, 3:5])
        ax5.plot(origin_data, label=['True CPU', 'True Mem'])
        start_x_pos = origin_data.shape[0] - len(compared_data['ours'])
        ax5.plot(range(start_x_pos, origin_data.shape[0]), compared_data['ours'],
                 label=['Predicted CPU', 'Predicted Mem'])
        ax5.axvline(x=start_x_pos, c='r', linestyle='--')
        ax5.set_title('(e) OURS', y=-0.25)
        ax5.set_ylabel("Utilization Rate/%")
        ax5.set_xlabel("Time/s")
        ax5.legend(fontsize=8)

        # 添加图片的辅助信息
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.95, hspace=0.3, wspace=0.5)
        plt.savefig('{}/{}_timeseries.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class MetrixFigure(Figure):
    def visual(self, machine_name, compared_data):
        """可视化各种损失函数"""

        acc = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}
        rmse = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}
        nrmse = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}
        nd = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}
        smape = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}

        for item in compared_data['dlinear']:
            acc['dlinear'].append(item['acc'])
            rmse['dlinear'].append(item['rmse'])
            nrmse['dlinear'].append(item['nrmse'])
            nd['dlinear'].append(item['nd'])
            smape['dlinear'].append(item['smape'])

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

        rmse_mean = {
            'arima': np.mean(rmse['arima']),
            'bht_arima': np.mean(rmse['bht_arima']),
            'lstm': np.mean(rmse['lstm']),
            'dlinear': np.mean(rmse['dlinear']),
            'ours': np.mean(rmse['ours'])
        }
        nrmse_mean = {
            'arima': np.mean(nrmse['arima']),
            'bht_arima': np.mean(nrmse['bht_arima']),
            'lstm': np.mean(nrmse['lstm']),
            'dlinear': np.mean(nrmse['dlinear']),
            'ours': np.mean(nrmse['ours'])
        }
        nd_mean = {
            'arima': np.mean(nd['arima']),
            'bht_arima': np.mean(nd['bht_arima']),
            'lstm': np.mean(nd['lstm']),
            'dlinear': np.mean(nd['dlinear']),
            'ours': np.mean(nd['ours'])
        }
        smape_mean = {
            'arima': np.mean(smape['arima']),
            'bht_arima': np.mean(smape['bht_arima']),
            'lstm': np.mean(smape['lstm']),
            'dlinear': np.mean(smape['dlinear']),
            'ours': np.mean(smape['ours'])
        }
        print(machine_name, '---> RMSE: ', rmse_mean, 'NRMSE: ', nrmse_mean, 'ND: ', nd_mean, 'SMAPE: ', smape_mean)

        # 创建一个 2 * 2 子图布局
        rows, cols = 2, 2
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))

        # 绘制第 1 个子图
        axs[0, 0].plot(rmse['arima'], label='ARIMA')
        axs[0, 0].plot(rmse['lstm'], label='LSTM')
        axs[0, 0].plot(rmse['bht_arima'], label='BHT ARIMA')
        axs[0, 0].plot(rmse['dlinear'], label='DLinear')
        axs[0, 0].plot(rmse['ours'], label='OURS')
        axs[0, 0].set_title('(a) RMSE', fontsize=11, y=-0.22)
        axs[0, 0].set_ylabel("RMSE")
        axs[0, 0].set_xlabel("Time/s")
        axs[0, 0].set_ylim([0, 100])
        axs[0, 0].legend(fontsize=8)

        # 绘制第 2 个子图
        axs[0, 1].plot(nrmse['arima'], label='ARIMA')
        axs[0, 1].plot(nrmse['lstm'], label='LSTM')
        axs[0, 1].plot(nrmse['bht_arima'], label='BHT ARIMA')
        axs[0, 1].plot(nrmse['dlinear'], label='DLinear')
        axs[0, 1].plot(nrmse['ours'], label='OURS')
        axs[0, 1].set_title('(b) Normalized RMSE', fontsize=11, y=-0.22)
        axs[0, 1].set_ylabel("Normalized RMSE")
        axs[0, 1].set_xlabel("Time/s")
        axs[0, 1].set_ylim([0, 2.5])
        axs[0, 1].legend(fontsize=8)

        # 绘制第 3 个子图
        axs[1, 0].plot(nd['arima'], label='ARIMA')
        axs[1, 0].plot(nd['lstm'], label='LSTM')
        axs[1, 0].plot(nd['bht_arima'], label='BHT ARIMA')
        axs[1, 0].plot(nd['dlinear'], label='DLinear')
        axs[1, 0].plot(nd['ours'], label='OURS')
        axs[1, 0].set_title('(c) Normalized Deviation', fontsize=11, y=-0.22)
        axs[1, 0].set_ylabel("Normalized Deviation")
        axs[1, 0].set_xlabel("Time/s")
        axs[1, 0].set_ylim([0, 3])
        axs[1, 0].legend(fontsize=8)

        # 绘制第 4 个子图
        axs[1, 1].plot(smape['arima'], label='ARIMA')
        axs[1, 1].plot(smape['lstm'], label='LSTM')
        axs[1, 1].plot(smape['bht_arima'], label='BHT ARIMA')
        axs[1, 1].plot(smape['dlinear'], label='DLinear')
        axs[1, 1].plot(smape['ours'], label='OURS')
        axs[1, 1].set_title('(d) SMAPE', fontsize=11, y=-0.22)
        axs[1, 1].set_ylabel("SMAPE")
        axs[1, 1].set_xlabel("Time/s")
        axs[1, 1].set_ylim([0, 2])
        axs[1, 1].legend(fontsize=8)

        # 添加整图标题
        fig.tight_layout()  # 调整子图布局以避免重叠
        plt.savefig('{}/{}_loss.png'.format(self.metrics_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()

        plt.figure()
        plt.plot(acc['arima'], label='ARIMA')
        plt.plot(acc['lstm'], label='LSTM')
        plt.plot(acc['bht_arima'], label='BHT ARIMA')
        plt.plot(acc['dlinear'], label='DLinear')
        plt.plot(acc['ours'], label='OURS')
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
        plt.title(job_name, y=-0.1, fontsize=20)  # 配置属性必须在 nx.draw 函数调用之前

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, font_color='whitesmoke', with_labels=True)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis("off")
        plt.savefig('{}/{}.png'.format(self.dag_saving_path, job_name),
                    format='png')
        plt.show()


class UDGFigure(Figure):
    def visual(self, G: Graph, udg_name=None):
        plt.title(udg_name, y=-0.1, fontsize=20)

        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, font_color='whitesmoke', with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

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
        axs[0].set_title('(a) Function Invoke Distribution', y=-0.25)
        axs[0].set_xlabel('Invoke times')
        axs[0].set_ylabel('Probability')
        axs[0].grid(linestyle='--', color='grey', alpha=0.3)

        y = s
        y_hat = s + (np.random.random(s.shape) - 0.5) * 5
        axs[1].scatter(x, y, marker='s', label='True Invoke Times')
        axs[1].scatter(x, y_hat, marker='o', label='Pred. Invoke Times')
        axs[1].set_title('(b) Function Invoke Frequencies', y=-0.25)
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

        fig = plt.figure(figsize=(10, 4))
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
        plt.ylabel('CPU Number', size='large')
        plt.xlabel('Execute Time/s', size='large')
        locs, labels = plt.yticks(pos, cpus)  # 重新设置 y 轴步长（locs）和每个步长对应的名称（labels）
        plt.setp(labels, fontsize=12)  # 设置 y 轴坐标
        ax.set_ylim(ymin=-0.1, ymax=num_cpus * 0.5 + 0.5)
        ax.set_xlim(xmin=-5)
        ax.grid(color='g', linestyle=':', alpha=0.75)

        plt.subplots_adjust(top=0.96, bottom=0.15, left=0.06, right=0.98)
        plt.savefig('{}/{}.png'.format(self.gantt_saving_path, self.timestamp),
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

        plt.subplots_adjust(top=0.96)
        plt.savefig('{}/{}.png'.format(self.schedule_results_saving_path, self.timestamp),
                    format='png')
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
            letter = chr(ord('a') + row * 3 + col)
            axs[row, col].set_title('(' + letter + ') ' + machine_name, fontsize=11, y=-0.5)
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
        plt.subplots_adjust(top=0.98, bottom=0.11)  # 调整整图标题的位置，以避免和子图重叠
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
        axs[0].set_title('(a) CPU Usage', y=-0.4, pad=5)
        axs[0].set_xlabel("Machine Name")
        axs[0].set_ylabel("Percent/%")

        axs[1].boxplot(all_mem_data)
        axs[1].set_xticklabels(x_axis_names, rotation=45)
        axs[1].set_title('(b) Memory Usage', y=-0.4, pad=5)
        axs[1].set_xlabel("Machine Name")
        axs[1].set_ylabel("Percent/%")

        fig.tight_layout()  # 调整子图布局以避免重叠
        plt.subplots_adjust(bottom=0.3)  # 调整整图标题的位置，以避免和子图重叠
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

        yy1 = [[0.428484848, 0.450909091, 0.48969697, 0.491818182, 0.512121212],
               [0.471415152, 0.514242424, 0.533030303, 0.542727273, 0.548181818],
               [0.679393939, 0.69030303, 0.70030303, 0.722121212, 0.758787879],
               [0.727272727, 0.74, 0.770606061, 0.784545455, 0.815454545],
               [0.75, 0.785151515, 0.796060606, 0.799090909, 0.813030303]]
        yy2 = [[0.553603604, 0.561261261, 0.562162162, 0.606306306, 0.613513514],
               [0.663288288, 0.668918919, 0.691666667, 0.709459459, 0.715315315],
               [0.728153153, 0.736936937, 0.747072072, 0.759684685, 0.769594595],
               [0.745945946, 0.753153153, 0.758108108, 0.802252252, 0.803153153],
               [0.755855856, 0.757657658, 0.767792793, 0.779279279, 0.843693694]]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].boxplot(yy1)
        axs[0].plot(x, np.mean(yy1, axis=1), 's--')
        axs[0].set_xticks(x, ['2', '3', '4', '5', '6'])
        axs[0].set_xlabel('Number of Layer(s)')
        axs[0].set_ylabel('Similarity')
        axs[0].set_title('(a) The Effect of Layer Numbers', y=-0.25)
        axs[0].grid(color='grey', linestyle=':', alpha=0.75)

        axs[1].boxplot(yy2)
        axs[1].plot(x, np.mean(yy2, axis=1), 's--')
        axs[1].set_xticks(x, ['0.1', '0.01', '1e-3', '1e-4', '1e-5'])
        axs[1].set_xlabel('Learning Rate')
        axs[1].set_ylabel('Similarity')
        axs[1].set_title('(b) The Effect of Learning Rate', y=-0.25)
        axs[1].grid(color='grey', linestyle=':', alpha=0.75)

        plt.tight_layout()
        filename = str(self.timestamp) + '_params'
        plt.savefig('{}/{}.png'.format(self.params_saving_path, filename),
                    dpi=600, format='png')
        plt.show()


class DQNParamsFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        x = [1, 2, 3, 4, 5]

        yy1 = [[1177.865613, 1276.284585, 1287.351779, 1296.837945, 1350.592885],
               [1123.715415, 1134.387352, 1134.782609, 1186.166008, 1251.778656],
               [923.715415, 945.4545455, 1031.225296, 1088.932806, 1096.442688],
               [891.6996047, 990.513834, 995.256917, 1024.901186, 1047.035573],
               [854.5454545, 873.5177866, 986.5612648, 1005.13834, 1006.719368]]
        yy2 = [[1255.212355, 1293.822394, 1296.525097, 1322.393822, 1397.297297],
               [1079.53668, 1128.571429, 1132.818533, 1190.733591, 1205.019305],
               [913.1274131, 928.957529, 989.96139, 1013.127413, 1074.517375],
               [928.1853282, 967.5675676, 1037.451737, 1057.915058, 1105.405405],
               [911.969112, 950.1930502, 956.3706564, 976.0617761, 1005.019305]]
        yy3 = [[1105.57377, 1117.04918, 1249.180328, 1282.622951, 1299.016393],
               [1053.114754, 1072.786885, 1079.016393, 1127.868852, 1162.295082],
               [939.0163934, 961.9672131, 1018.360656, 1050.491803, 1067.540984],
               [888.852459, 923.6065574, 955.7377049, 1065.245902, 1073.114754],
               [931.4754098, 948.852459, 974.0983607, 1028.852459, 1039.672131]]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].boxplot(yy1)
        axs[0].plot(x, np.mean(yy1, axis=1), 's--')
        axs[0].set_xticks(x, ['1024', '2048', '4096', '8192', '16384'])
        axs[0].set_xlabel('Experience Replay Buffer Size')
        axs[0].set_ylabel('Average Makespan/ms')
        axs[0].set_title('(a) The Effect of Replay Buffer Size', y=-0.25)
        axs[0].grid(color='grey', linestyle=':', alpha=0.75)

        axs[1].boxplot(yy2)
        axs[1].plot(x, np.mean(yy2, axis=1), 's--')
        axs[1].set_xticks(x, ['0.1', '0.01', '1e-3', '1e-4', '1e-5'])
        axs[1].set_xlabel('Learning Rate')
        axs[1].set_ylabel('Average Makespan/ms')
        axs[1].set_title('(b) The Effect of Learning Rate', y=-0.25)
        axs[1].grid(color='grey', linestyle=':', alpha=0.75)

        axs[2].boxplot(yy3)
        axs[2].plot(x, np.mean(yy3, axis=1), 's--')
        axs[2].set_xticks(x, ['16', '32', '64', '128', '256'])
        axs[2].set_xlabel('Batch Size')
        axs[2].set_ylabel('Average Makespan/ms')
        axs[2].set_title('(c) The Effect of Batch Size', y=-0.25)
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

        axs[0].plot(problem_size, makespan[:, 0], '^--', label='DPE')
        axs[0].plot(problem_size, makespan[:, 1], '*--', label='HEFT')
        axs[0].plot(problem_size, makespan[:, 2], 's--', label='OURS')
        axs[0].set_xticklabels(problem_size, rotation=45)
        axs[0].set_title('(a) The Effect of Problem Size on Makespan', y=-0.4)
        axs[0].set_xlabel('Problem Size')
        axs[0].set_ylabel('Makespan/ms')
        axs[0].legend()

        axs[1].plot(problem_size, runtime[:, 0], '^--', label='DPE')
        axs[1].plot(problem_size, runtime[:, 1], '*--', label='HEFT')
        axs[1].plot(problem_size, runtime[:, 2], 's--', label='OURS')
        axs[1].set_xticklabels(problem_size, rotation=45)
        axs[1].set_title('(b) The Effect of Problem Size on Runtime', y=-0.4)
        axs[1].set_ylim([0, 20])
        axs[1].set_xlabel('Problem Size')
        axs[1].set_ylabel('Runtime/s')
        axs[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.29)
        plt.savefig('{}/{}_problem_size.png'.format(self.schedule_results_saving_path, self.timestamp),
                    dpi=600, format='png')
        plt.show()


class ChineseTextFigure(Figure):
    def visual(self, origin_data=None, compared_data=None):
        x = [1, 2, 3, 4, 5]
        y = [0.2, 0.4, 0.6, 0.2, 0.5]
        plt.plot(x, y, label='折线')
        plt.xlabel('横坐标')
        plt.ylabel('纵坐标')
        plt.title('宋体 —— Times New Roman —— $E=mc^2$')
        plt.legend()
        plt.show()
