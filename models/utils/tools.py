"""工具类"""
import random


def generate_random_numbers(count):
    """生成 count 个不重复的随机数，总和为 1"""
    numbers = random.sample(range(1, 100), count)
    total = sum(numbers)
    return [round(number / total, 2) for number in numbers]
