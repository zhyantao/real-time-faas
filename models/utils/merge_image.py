"""拼接图像"""
import math
import time

from PIL import Image

from models.utils.params import args


def merge(image_list):
    n = len(image_list)

    # 设置拼接后图像的行列数
    rows, cols = math.ceil(n / 3), 3

    # 打开要拼接的 PNG 图像
    images = [Image.open(image_path) for image_path in image_list]

    # 获取单个图像的大小
    width, height = images[0].size

    # 创建一个新的图像，用于存储拼接后的图像
    result = Image.new('RGBA', (cols * width, rows * height))

    # 拼接图像
    for i in range(rows):
        for j in range(cols):
            result.paste(images[i * cols + j], (j * width, i * height))

    # 保存拼接后的图像
    result.save(args.result_saving_path + '/{}_merged_image.png'.format(time.time()))


if __name__ == '__main__':
    image_list1 = [
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_11624.png",
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_12288.png",
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_34819.png",
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_54530.png",
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_77773.png",
        "E:\\Workshop\\real-time-faas\\results\\dags\\j_82634.png"
    ]
    merge(image_list1)
