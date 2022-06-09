import os
import paddlex as pdx
from paddlex import transforms as T  # paddlex >= 2.0.0


def build_model():
    # 设置 GPU 训练，如果没有 GPU 则使用 CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 数据增强
    train_transforms = T.Compose([
        T.MixupImage(mixup_epoch=250),
        T.RandomDistort(),
        T.RandomExpand(),
        T.RandomCrop(),
        T.Resize(target_size=608, interp='RANDOM'),
        T.RandomHorizontalFlip(),
        T.Normalize(),
    ])

    eval_transforms = T.Compose([
        T.Resize(target_size=608, interp='CUBIC'),
        T.Normalize(),
    ])

    # 定义数据集
    train_dataset = pdx.datasets.VOCDetection(
        data_dir='data/train',
        file_list='data/train/train_list.txt',
        label_list='data/train/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir='data/train',
        file_list='data/train/val_list.txt',
        label_list='data/train/labels.txt',
        transforms=eval_transforms)

    # 正式训练
    num_classes = len(train_dataset.labels)
    model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
    model.train(
        num_epochs=100,
        train_dataset=train_dataset,
        train_batch_size=2,
        eval_dataset=eval_dataset,
        learning_rate=0.000125,
        lr_decay_epochs=[210, 240],
        save_interval_epochs=20,
        save_dir='output/yolov3_darknet53',
        use_vdl=True)
