import matplotlib.pyplot as plt


class Figure:
    def __init__(self):
        pass

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

        # 添加图片的辅助信息
        plt.suptitle('CPU and Mem Usage Prediction')
        plt.ylabel("Utilization Rate (%)")
        plt.xlabel("Relative Time (s)")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    x1 = [0, 1, 2, 3, 4]
    y1 = [1, 2, 2, 6, 8]

    x2 = [2, 3, 4, 5, 6]
    y2 = [3, 4, 2, 6, 4]

    plt.plot(x1, y1, '-o', label='Line 1')
    plt.plot(x2, y2, '-o', label='Line 2')
    plt.legend()
    plt.show()
