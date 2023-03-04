import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


def tsplot(y, lags=None, figsize=(12, 7)):
    """
    1）绘制时间序列曲线，并计算它的自相关（ACF）和部分自相关系数（PACF）。
    2）计算增强 Dickey-Fuller 测试。

    自相关系数可以用来判断时间序列是不是平稳的，这也有助于找到 ARIMA 模型中移动平均部分的阶数。
    部分自相关系数有助于确定 ARIMA 模型中自回归部分的阶数。
    增强 Dickey-Fuller 单元测试检查时间序列是否是非平稳的。
    零假设是序列是非平稳的，因此如果 p 值很小，则意味着时间序列不是非平稳的。

    Plot time series, its ACF and PACF, calculate Dickey–Fuller test

    y - timeseries
    lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()


tsplot(ts_sun)

ts_sun_diff = (ts_sun - ts_sun.shift(1)).dropna()
tsplot(ts_sun_diff)


