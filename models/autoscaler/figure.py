"""论文插图"""

import os.path
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx import DiGraph

from models.autoscaler.dag import DAG
from models.utils.dataset import get_one_job
from models.utils.params import args


class Figure:
    def __init__(self):

        self.result_saving_path = args.result_saving_path
        self.metrics_saving_path = self.result_saving_path + '/metrics'
        self.dag_saving_path = self.result_saving_path + '/dags'

        self.timestamp = time.time()  # 用于区分不同时刻产生的结果文件

        # 检查是否存在保存实验结果的路径
        if not os.path.exists(self.result_saving_path):
            os.makedirs(self.result_saving_path)
        if not os.path.exists(self.metrics_saving_path):
            os.makedirs(self.metrics_saving_path)
        if not os.path.exists(self.dag_saving_path):
            os.makedirs(self.dag_saving_path)

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


if __name__ == '__main__':
    # 生成数据
    x1 = np.random.rand(50)
    y1 = np.random.rand(50)
    x2 = np.random.rand(50)
    y2 = np.random.rand(50)

    # 绘制折线图
    plt.figure()
    plt.plot(x1, y1, '-o', label='Line 1')
    plt.plot(x2, y2, '-o', label='Line 2')
    plt.legend()
    plt.show()

    # 绘制散点图
    plt.figure()
    plt.scatter(x1, y1, color='red', marker='o')
    plt.scatter(x2, y2, color='blue', marker='s')
    plt.title('Multiple Scatter Plots')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()

    # 重建 DAG
    df = pd.read_csv(args.selected_batch_task_path)
    job, idx = get_one_job(df)
    G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
    dag_figure = DAGFigure()
    dag_figure.visual(G, job_name)
    job, idx = get_one_job(df, idx)
    G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
    dag_figure = DAGFigure()
    dag_figure.visual(G, job_name)
