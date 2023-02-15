import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from models.utils.dataset import *


class MyARIMA:

    def __init__(self, order, future_periods):
        self.future_periods = future_periods
        self.order = order

    def gauss_compare(self, original_series, predictions, data_split):
        # the train/test split used to generate the Gaussian-filtered predictions
        size = int(len(original_series) * data_split)

        # creating a plot of the original series and Gaussian-filtered predictions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        myFmt = mdates.DateFormatter('%m/%y')
        ax.xaxis.set_major_formatter(myFmt)

        plt.plot(original_series[size:])
        plt.plot(predictions, color='red')
        plt.title('Gauss-Filtered Predictions vs. Original Series')
        plt.show()

        # calculating the RMSE between the Gaussian-filtered predictions and original dataset.
        # the +1 exception code is required when differencing is performed, as the earliest data point can be lost
        try:
            error = np.sqrt(mean_squared_error(predictions, original_series[size:]))
        except:
            error = np.sqrt(mean_squared_error(predictions, original_series[size + 1:]))
        print('Test RMSE: %.3f' % error)

    def predict(self, selected_container_usage_path=para.get("selected_container_usage_path")):
        """
        使用 ARIMA 模型预测资源需求

        :param selected_container_usage_path:
        :return:
        """
        if not os.path.exists(selected_container_usage_path):
            print("container_usage.csv has not been selected.")

        df = pd.read_csv(selected_container_usage_path)
        rows = df.shape[0]
        idx = 0

        split_rate = 0.67

        while idx < rows:
            machine, idx = get_one_machine(df, idx)

            training_data_cpu = machine.iloc[:, 3:4].values  # CPU
            training_data_mem = machine.iloc[:, 4:5].values  # Mem

            train_size = int(len(training_data_cpu) * split_rate)
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

            # # 基于用户输入的 future_periods 预测 test_cpu set 之后的一段时间
            predictions_cpu = pd.Series(predictions_cpu)
            predictions_mem = pd.Series(predictions_mem)

            plt.axvline(x=train_size, c='r', linestyle='--')  # 分割线

            plt.plot(np.append(train_cpu, test_cpu), label='Real CPU Util.')  # 真实值
            plt.plot(np.append(train_cpu, predictions_cpu), label='Predicted CPU Util.')  # 预测值
            plt.plot(np.append(train_mem, test_mem), label='Real Mem Util.')
            plt.plot(np.append(train_mem, predictions_mem), label='Predicted Mem Util.')
            plt.suptitle("CPU Usage Prediction")
            plt.ylabel("Utilization Rate (%)")
            plt.xlabel("Relative Time (s)")
            plt.legend()
            plt.show()

            # calculates root mean squared errors (RMSEs) for the out-of-sample predictions_cpu
            error_cpu = np.sqrt(mean_squared_error(predictions_cpu, test_cpu))
            error_mem = np.sqrt(mean_squared_error(predictions_mem, test_mem))
            print('Test RMSE (CPU): %.3f' % error_cpu)
            print('Test RMSE (Mem): %.3f' % error_mem)
