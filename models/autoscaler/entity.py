"""实体类"""

from queue import Queue


class Task:
    def __init__(self, duration, queue_size=50):
        self.start_time_queue = Queue(queue_size)
        self.end_time_queue = Queue(queue_size)
        self.mem_usage = Queue(queue_size)
        self.cpu_usage = Queue(queue_size)
        self.calling_times_self = Queue(queue_size)
        self.calling_times_children = Queue(queue_size)
        self.duration = duration


class Node:
    def __init__(self):
        pass
