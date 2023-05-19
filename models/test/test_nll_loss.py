import torch
import torch.nn.functional as F


def test_nll_loss():
    # 模型输出，假设有 10 个类别
    output = torch.randn(3, 10)

    # 真实标签，假设为第 3、5、7 个类别
    target = torch.tensor([2, 4, 6])

    # 对模型输出进行 softmax 操作
    output = F.softmax(output, dim=1)

    # 计算损失
    loss = F.nll_loss(output, target)

    print(output)
    print(target)
    print(loss)
