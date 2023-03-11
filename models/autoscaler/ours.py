class Ours:
    """本文提出的资源预测算法"""

    def __init__(self):
        pass

    def merge(self, arima_output, lstm_output):
        return (arima_output + lstm_output) / 2.0
