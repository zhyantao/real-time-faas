import matplotlib.pyplot as plt
import pandas as pd

from models.utils.dataset import get_one_machine
from models.utils.params import args


def get_workload():
    df = pd.read_csv(args.selected_container_usage_path)
    rows = df.shape[0]
    idx, count = 0, 1
    while idx < rows:
        machine, idx = get_one_machine(df, idx)
        # print(machine)

        # 处理单个 machine 的代码
        ori_data = machine.iloc[:, 3:5].values
        plt.plot(ori_data, label=['CPU Usage', 'Mem Usage'])
        plt.legend()
        plt.show()

        count += 1
        if count == 6:
            break


if __name__ == '__main__':
    get_workload()
