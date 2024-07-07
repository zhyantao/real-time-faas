from models.utils.dataset import *


def test_generate_dataset():
    """
    step 1: 由于 alibaba 数据集过于庞大，本文仅使用部分数据集作为本文的输入。
    :return:
    """
    download_batch_task()
    sample_jobs()
    sample_machines()


def test_generate_inputs():
    """
    step 2: 根据提取出来的数据集，进一步将其整理为可直接作为实验输入的数据。
    :return:
    """
    gen_task_depend_input()
    gen_node_connect_input()

if __name__ == '__main__':
    test_generate_dataset()
    test_generate_inputs()