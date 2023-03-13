import matplotlib.pyplot as plt
import numpy as np


def runtime_distribution_figure():
    x = np.linspace(0.1, 10, 1000)
    y = np.log(x)  # 总运行时间花费
    y2 = x  # 系统资源花费

    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    for i in range(2):
        for j in range(3):
            axs[i, j].plot(x, y)
            axs[i, j].plot(x, y2)
            axs[i, j].set_title(str(i) + ', ' + str(j))
            axs[i, j].set_xlabel('invoke times')
            axs[i, j].set_ylabel('time cost (s)')

    plt.tight_layout()
    plt.show()


def invoke_simulation_figure():
    # 采样点数
    n_points = 100  # 模拟 1000 次调用

    # 设置均值和标准差
    mu = 0
    sigma = 1

    # 生成包含100个随机数的正态分布序列
    x = np.linspace(0, 20 * 60 * 1000, n_points)  # 20 分钟
    y = np.random.normal(mu, sigma, n_points)

    plt.scatter(x, y)
    plt.show()

    # 打印序列
    print(y)


def branch_prediction_miss_figure():
    # 设置随机数种子，以便多次运行时生成相同的曲线
    np.random.seed(42)

    # 生成 x 值
    x = np.linspace(0, 10, 100)

    # 生成 y 值，其中包含随机噪声
    y1 = 10 - x + np.random.normal(0, 1, 100)
    y2 = 20 - x + np.random.normal(0, 1, 100)
    y3 = 5 - x + np.random.normal(0, 1, 100)

    # 创建一个新的图形
    plt.figure()

    # 绘制曲线
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)

    # 添加标题和标签
    plt.title('Branch prediction misses')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    invoke_simulation_figure()
    branch_prediction_miss_figure()
    runtime_distribution_figure()
