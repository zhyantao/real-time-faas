"""LSTM 多步预测"""

import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable

from models.utils.dataset import *


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

    def sliding_windows(self, data):
        x = []
        y = []

        for i in range(len(data) - self.seq_length - 1):
            _x = data[i:i + self.seq_length]  # 每次截取一段
            _y = data[i + self.seq_length]  # 每次拼接一个
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def predict(self, selected_container_usage_path=para.get("selected_container_usage_path")):
        """
        测试选中的 container usage 预测效果

        :param selected_container_usage_path:
        :return:
        """
        if not os.path.exists(selected_container_usage_path):
            print("container_usage.csv has not been selected.")

        df = pd.read_csv(selected_container_usage_path)
        rows = df.shape[0]
        idx = 0

        while idx < rows:
            machine, idx = get_one_machine(df, idx)
            # print(machine)

            # 处理单个 machine 的代码
            training_set = machine.iloc[:, 3:5].values
            # plt.plot(training_set, label="cpu_util_percent")
            # plt.show()

            # 归一化处理
            sc = MinMaxScaler()
            training_data = sc.fit_transform(training_set)

            x, y = self.sliding_windows(training_data)
            # print('x: ', x)
            # print('y: ', y)

            train_size = int(len(y) * 0.67)
            test_size = len(y) - train_size

            data_x = Variable(torch.Tensor(np.array(x)))
            data_y = Variable(torch.Tensor(np.array(y)))

            train_x = Variable(torch.Tensor(np.array(data_x[0:train_size])))
            train_y = Variable(torch.Tensor(np.array(data_y[0:train_size])))

            test_x = Variable(torch.Tensor(np.array(data_x[train_size:len(x)])))
            test_y = Variable(torch.Tensor(np.array(data_y[train_size:len(y)])))

            num_epochs = 2000
            learning_rate = 0.01

            criterion = torch.nn.MSELoss()  # mean-squared error for regression
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

            # Train the model
            for epoch in range(num_epochs):
                outputs = self(train_x)
                optimizer.zero_grad()

                # obtain the loss function
                loss = criterion(outputs, train_y)

                loss.backward()

                optimizer.step()
                # if epoch % 100 == 0:
                #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

            self.eval()  # 评估模型
            train_predict = self(test_x)  # 预测参数

            data_predict = train_predict.data.numpy()
            data_y_plot = test_y.data.numpy()

            data_predict = sc.inverse_transform(data_predict)
            data_y_plot = sc.inverse_transform(data_y_plot)

            plt.axvline(x=train_size, c='r', linestyle='--')

            plt.plot(data_y_plot, label=['Real CPU Util.', 'Real Mem Util.'])
            plt.plot(data_predict, label=['Predicted CPU Util.', 'Predicted Mem Util.'])
            plt.suptitle('CPU and Mem Usage Prediction')
            plt.ylabel("Utilization Rate (%)")
            plt.xlabel("Relative Time (s)")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    lstm = LSTM(num_classes=2, input_size=2, hidden_size=2, num_layers=1, seq_length=4)
    lstm.predict()
