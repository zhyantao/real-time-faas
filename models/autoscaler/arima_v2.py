from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA

from models.utils.dataset import *


class ARIMA_v2:

    def __init__(self, order, future_periods):
        self.future_periods = future_periods
        self.order = order

    def predict(self, machine: DataFrame, train_size):
        """使用 ARIMA 模型预测资源需求"""
        training_data_cpu = machine.iloc[:, 3:4].values  # CPU
        training_data_mem = machine.iloc[:, 4:5].values  # Mem

        train_size = train_size
        test_size = len(training_data_cpu) - train_size

        train_cpu = training_data_cpu[0:train_size]
        test_cpu = training_data_cpu[train_size:len(training_data_cpu)]
        train_mem = training_data_mem[0:train_size]
        test_mem = training_data_mem[train_size:len(training_data_mem)]

        history_cpu = [val for val in train_cpu]
        history_mem = [val for val in train_mem]

        predictions_cpu = []
        predictions_mem = []

        # 1) 通过 test_cpu set 中的一个值创建滚动预测
        # 2) 将这个值添加到 model_cpu 中进行训练
        # 3) 回到步骤 1 预测 test_cpu set 中的下一个值
        for t in range(len(test_cpu)):
            model_cpu = ARIMA(history_cpu, order=(self.order[0], self.order[1], self.order[2]))
            model_mem = ARIMA(history_mem, order=(self.order[0], self.order[1], self.order[2]))

            model_fit_cpu = model_cpu.fit()
            model_fit_mem = model_mem.fit()

            output_cpu = model_fit_cpu.forecast()
            output_mem = model_fit_mem.forecast()
            yhat_cpu = output_cpu[0]
            yhat_mem = output_mem[0]

            predictions_cpu.append(yhat_cpu)
            predictions_mem.append(yhat_mem)

            obs_cpu = test_cpu[t]
            obs_mem = test_mem[t]

            history_cpu.append(obs_cpu)
            history_mem.append(obs_mem)

        # 基于用户输入的 future_periods 预测 test_cpu set 之后的一段时间
        predictions_cpu = pd.Series(predictions_cpu)
        predictions_mem = pd.Series(predictions_mem)

        return predictions_cpu, predictions_mem
