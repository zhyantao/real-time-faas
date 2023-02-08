"""
可视化工具
"""
import sys

import matplotlib.pyplot as plt


class ProgressBar:
    """
    进度条
    """

    def __init__(self, width=50):
        self.last = -1
        self.width = width

    def update(self, current):
        assert 0 <= current <= 100
        if self.last == current:
            return
        self.last = int(current)
        pos = int(self.width * (current / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(current), '#' * pos + '.' * (self.width - pos)))
        sys.stdout.flush()
        if current == 100:
            print('')


class LineChart:
    """
    折线图
    """

    def __init__(self, x_data, y_data, title, label, x_label, y_label, legend_loc="lower right"):
        self.x_data = x_data  # 一维数组
        self.y_data = y_data  # 一维数组
        self.title = title  # 图名
        self.label = label  # 图例名称
        self.x_label = x_label  # x 轴名称
        self.y_label = y_label
        self.legend_loc = legend_loc  # 图例位置

    def show(self):
        x = self.x_data
        y = self.y_data

        plt.plot(x, y, "g", label=self.label)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend(loc=self.legend_loc)  # 设置图例的放置位置

        # # 在图上标注数字
        # for xx, yy in zip(x, y):
        #     plt.text(xx, yy, str(yy), ha="center", va="bottom", fontsize=10)

        plt.show()
