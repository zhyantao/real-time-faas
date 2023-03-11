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


bar = ProgressBar()
