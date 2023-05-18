import os

from invoker import *
from models.autoscaler.arima_v2 import ARIMA_v2
from models.autoscaler.lstm import LSTM
from models.autoscaler.metrics import metrics
from models.utils.figure import TimeSeriesFigure, MetrixFigure
from models.utils.parameters import args
from models.utils.text import ProgressBar
from models.utils.tools import get_one_machine

if __name__ == '__main__':
    selected_container_usage_path = args.selected_container_usage_path

    if not os.path.exists(selected_container_usage_path):
        print("container_usage.csv has not been selected.")

    df = pd.read_csv(selected_container_usage_path)
    rows = df.shape[0]

    idx, count = 0, 0
    while idx < rows:
        # (1) 每次从文件中读取一个 task 的资源需求变化
        machine, next_idx = get_one_machine(df, idx)
        print(machine)
        machine_name = machine['machine_id'].loc[machine.index[0]]
        print(machine_name)

        # training_data_cpu = machine.iloc[:, 3:4].values  # CPU
        # training_data_mem = machine.iloc[:, 4:5].values  # memory
        training_data = machine.iloc[:, 3:5].values  # CPU 和 memory

        predictions = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}  # 统计预测值
        losses = {'arima': [], 'bht_arima': [], 'lstm': [], 'dlinear': [], 'ours': []}  # 统计损失

        train_size = int(training_data.shape[0] * 0.37)  # 用 37% 的数据预测前面的数据

        # 调用 DLinear 模型
        y_hat_dlinear, y_test_dlinear = run_dlinear(idx)
        # print("y_hat_dlinear --> ", y_hat_dlinear.shape, y_hat_dlinear)
        # print("y_test_dlinear --> ", y_test_dlinear.shape, y_test_dlinear)
        for i in range(y_hat_dlinear.shape[0]):
            predictions['dlinear'].append(y_hat_dlinear[i])
            y = training_data[train_size + i].reshape(-1, 1)
            losses['dlinear'].append(metrics(y_hat_dlinear[i], y))

        # 调用 ARIMA 预测模型
        p, d, q = 2, 1, 2
        params = [p, d, q]
        future_periods = 12
        my_arima = ARIMA_v2(params, future_periods)
        predictions_cpu, predictions_mem = my_arima.predict(machine, train_size)
        for i in range(predictions_cpu.shape[0]):
            y_hat_arima = np.array([predictions_cpu[i], predictions_mem[i]])
            y = training_data[train_size + i].reshape(-1, 1)
            predictions['arima'].append(y_hat_arima)
            losses['arima'].append(metrics(y_hat_arima, y))

        # 调用 LSTM 模型
        seq_length = 4
        lstm = LSTM(num_classes=2, input_size=2, hidden_size=128, num_layers=1, seq_length=seq_length)
        y_hat_lstm = lstm.predict(machine, train_size - seq_length - 1)
        for i in range(y_hat_lstm.shape[0]):
            y = training_data[train_size + i].reshape(-1, 1)
            predictions['lstm'].append(y_hat_lstm[i])
            losses['lstm'].append(metrics(y_hat_lstm[i], y))

        # 多步预测，有些离谱
        # y_hat_lstm_v2, y_test_lstm_v2 = run_lstm_v2_multi_step(training_data.T, train_size)
        # print(y_hat_lstm_v2.shape)
        # print(training_data.shape)
        # start_x = training_data.shape[0] - y_hat_lstm_v2.shape[0]
        # for i in range(y_hat_lstm_v2.shape[0]):
        #     y = training_data[train_size + i + seq_length].reshape(-1, 1)
        #     predictions['ours'].append(y_hat_lstm_v2[i])
        #     losses['ours'].append(metrics(y_hat_lstm_v2[i], y))

        # 使用本文提出的模型
        n_samples = training_data.shape[0]
        bar = ProgressBar()
        for i in range(train_size, n_samples):
            # (2) 准备数据
            X = training_data[i - train_size:i].T
            y = training_data[i].reshape(-1, 1)

            # 调用 BHT ARIMA 模型
            y_hat_bht_arima = run_bht_arima(X, y)
            predictions['bht_arima'].append(y_hat_bht_arima)
            losses['bht_arima'].append(metrics(y_hat_bht_arima, y))

            # (3) 调用 Ours 预测模型
            y_hat_ours = run_lstm_v2(X, y)
            predictions['ours'].append(y_hat_ours)
            losses['ours'].append(metrics(y_hat_ours, y))
            # print('y_hat_ours = {}'.format(y_hat_ours))
            # print("evaluation (Ours): {}\n".format(metrics(y_hat_ours, y)))

            bar.update(percent=100.0 * (i - train_size) / (n_samples - train_size - 1))

        ts_figure = TimeSeriesFigure()
        ts_figure.visual(training_data, predictions)
        loss_figure = MetrixFigure()
        loss_figure.visual(machine_name, losses)

        print('-------------- sample {} end ----------------'.format(count))
        idx = next_idx
        count += 1
