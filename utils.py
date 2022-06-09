import os
import time
import random
import paddlex as pdx
import numpy as np
import pandas as pd


def load_data():
    """
    本次实验用到的数据集需要手动下载并解压到 data 目录下，
    形成 data/train 和 data/test 两个文件夹。

    - train.zip：https://zhyantao.lanzouj.com/iBAxI00h3vzc
    - test.zip：https://zhyantao.lanzouj.com/imHNM00h3vcj
    """
    pass


def build_dataset():
    build_mapper()


def build_transformer():
    pass


def build_mapper():
    # 遍历训练集
    name = [name for name in os.listdir('data/train/IMAGES') if name.endswith('.jpg')]

    train_name_list = []
    for i in name:
        tmp = os.path.splitext(i)
        train_name_list.append(tmp[0])

    # 构造图片 -xml 的链接文件 ori_train.txt
    with open("data/train/ori_train.txt", "w") as f:
        for i in range(len(train_name_list)):
            if i != 0:
                f.write('\n')
            line = 'IMAGES/' + train_name_list[i] + '.jpg' + " " + "ANNOTATIONS/" + train_name_list[i] + '.xml'
            f.write(line)

    # 构造 label.txt
    labels = ['crazing', 'inclusion', 'pitted_surface', 'scratches', 'patches', 'rolled-in_scale']
    with open("data/train/labels.txt", "w") as f:
        for i in range(len(labels)):
            line = labels[i] + '\n'
            f.write(line)

    # 将 ori_train.txt 随机按照 eval_percent 分为验证集文件和训练集文件
    # eval_percent 验证集所占的百分比

    eval_percent = 0.2

    data = []
    with open("data/train/ori_train.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line)

    index = list(range(len(data)))
    random.shuffle(index)

    # 构造验证集文件
    cut_point = int(eval_percent * len(data))
    with open("data/train/val_list.txt", "w") as f:
        for i in range(cut_point):
            if i != 0:
                f.write('\n')
            line = data[index[i]]
            f.write(line)

    # 构造训练集文件
    with open("data/train/train_list.txt", "w") as f:
        for i in range(cut_point, len(data)):
            if i != cut_point:
                f.write('\n')
            line = data[index[i]]
            f.write(line)


def visualization():
    # 结果可视化
    model = pdx.load_model('output/yolov3_darknet53/best_model')

    for index in range(1550, 1800):
        image_name = 'data/test/IMAGES/' + str(index) + '.jpg'
        predicts = model.predict(image_name)
        pdx.det.visualize(image_name, predicts, threshold=0.1, save_dir='output/yolov3_darknet53/testImage/')


def submit():
    name = [name for name in os.listdir('data/test/IMAGES') if name.endswith('.jpg')]
    test_name_list = []
    for i in name:
        tmp = os.path.splitext(i)
        test_name_list.append(tmp[0])

    # 读取模型
    model = pdx.load_model('output/yolov3_darknet53/best_model')

    # 建立一个标号和题目要求的 id 的映射
    num2index = {
        'crazing': 0,
        'inclusion': 1,
        'pitted_surface': 2,
        'scratches': 3,
        'patches': 4,
        'rolled-in_scale': 5
    }

    result_list = []
    # 将置信度较好的框写入 result_list
    for index in test_name_list:
        image_name = 'data/test/IMAGES/' + index + '.jpg'
        predicts = model.predict(image_name)
        for predict in predicts:
            # print(predict['score'])
            if predict['score'] < 0.1:
                continue
            # 将 bbox 转化为题目中要求的格式
            tmp = predict['bbox']
            tmp[2] += tmp[0]
            tmp[3] += tmp[1]
            line = [index, tmp, num2index[predict['category']], predict['score']]
            result_list.append(line)

    # 将 result_list 写入 csv 文件，用于提交比赛结果
    result_array = np.array(result_list, dtype='object')
    # print(result_array)
    df = pd.DataFrame(result_array, columns=['image_id', 'bbox', 'category_id', 'confidence'])
    df.to_csv('submit/submission' + str(time.time()) + '.csv', index=False)
