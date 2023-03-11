"""工具类"""

import random
import sys


class ProgressBar:
    """进度条"""

    def __init__(self, width=50):
        self.last = -1
        self.width = width

    def update(self, percent):
        assert 0 <= percent <= 100
        if self.last == percent:
            return
        self.last = int(percent)
        pos = int(self.width * (percent / 100.0))
        sys.stdout.write('\r%d%% [%s]' % (int(percent), '#' * pos + '.' * (self.width - pos)))
        sys.stdout.flush()
        if percent == 100:
            print('')


def generate_random_numbers(count):
    # 生成 count 个不重复的随机数，总和为 1
    numbers = random.sample(range(1, 100), count)
    total = sum(numbers)
    return [round(number / total, 2) for number in numbers]
