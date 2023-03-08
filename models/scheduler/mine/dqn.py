import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


class ConvNet(nn.Module):
    """ConvNet 模型"""

    def __init__(self, input_shape, output_shape):
        """
        初始化模型的各个层
        :param input_shape:
        :param output_shape:
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 手动计算池化层的大小
        self.drop1 = nn.Dropout2d(p=0.2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=16 * (input_shape[0] // 2) * (input_shape[1] // 2), out_features=256)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=output_shape)

    def forward(self, x):
        """
        定义模型的前向传播过程
        :param x:
        :return:
        """
        # print('dqn.py --> x.shape (input): ')
        # print(x.shape)
        # print('dqn.py --> x.shape (input) end')
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        # print('dqn.py --> x.shape (output): ')
        # print(x.shape)
        # print('dqn.py --> x.shape (output) end')
        return x


class DQN(nn.Module):
    """DQN Implementation."""

    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.lr = 0.01
        self.gamma = 1
        self.batch_size = 32
        self.min_experiences = 100
        self.max_experiences = 10000
        self.num_actions = output_shape
        self.model = ConvNet(input_shape, output_shape)  # 核心还是 CNN 模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def predict(self, input_data):
        """
        预测给定状态下的 Q 值
        :param input_data:
        :return:
        """
        # print('dqn.py --> input_data.shape: ')
        # print(input_data.shape)
        # print('dqn.py --> input_data.shape end')
        # torch.Size([1, 10, 400, 1]) 表示 4 维张量：
        # - 第一维大小是 1
        # - 第二维大小是 10
        # - 第三维大小是 100
        # - 第四维大小是 1
        # 可以理解为一个 1 个样本，10 个通道，每个通道的特征有 400 个维度，每个维度的值为 1。
        # 这个形状的张量常用于表示一个 batch 大小为 1 的卷积神经网络的输入或输出，其中第一维度表示 batch 的大小，
        # 第二维度表示通道数，第三维度和第四维度分别表示每个通道的特征图的高度和宽度。
        # print('dqn.py --> input_data: ')
        # print(input_data)
        # print('dqn.py --> input_data end')
        return self.model(
            torch.tensor(
                input_data.float().reshape(
                    input_data.shape[0],  # batch_size
                    1,  # channels
                    input_data.shape[1],  # height
                    input_data.shape[2]  # width
                )
            )
        )

    def _train(self, dqn_target):
        """
        训练 DQN
        :param dqn_target:
        :return:
        """
        if len(self.experience['s']) < self.min_experiences:
            return

        # samples
        ids = torch.randint(low=0, high=len(self.experience['s']), size=(self.batch_size,))
        states = torch.tensor([self.experience['s'][i] for i in ids])
        actions = torch.tensor([self.experience['a'][i] for i in ids])
        rewards = torch.tensor([self.experience['r'][i] for i in ids])
        states_next = torch.tensor([self.experience['s2'][i] for i in ids])
        dones = torch.tensor([self.experience['done'][i] for i in ids])

        # print('dqn.py --> state_next: ')
        # print(states_next)  # [batch_size, height, width] = [32, 10, 100]
        # print('dqn.py --> state_next end')

        # 使用目标网络计算真实的值
        # print('dqn.py --> dqn_target.predict(states_next): ')
        # print(dqn_target.predict(states_next))
        # print('dqn.py --> dqn_target.predict(states_next) end')
        values_next = torch.max(dqn_target.predict(states_next), dim=1)[0]
        # print('dqn.py --> values_next: ')
        # print(values_next)
        # print('dqn.py --> values_next end')
        actual_values = torch.where(dones, rewards, rewards + self.gamma * values_next)

        # 使用训练的模型，预测新值和计算损失
        predicted_values = torch.sum(
            self.predict(states) * F.one_hot(actions, self.num_actions), dim=1)
        loss = torch.sum(torch.square(actual_values - predicted_values))

        # 应用梯度下降更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, states, epsilon):
        """
        根据状态预测动作
        :param states:
        :param epsilon:
        :return:
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return torch.argmax(self.predict(torch.tensor([states]))[0])

    def add_experience(self, exp):
        """
        向重放缓存中加入经验
        :param exp:
        :return:
        """
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, dqn_src):
        """
        在模型见复制权重
        :param dqn_src:
        :return:
        """
        variables1 = self.model.parameters()
        variables2 = dqn_src.model.parameters()
        for v1, v2 in zip(variables1, variables2):
            v1.data.copy_(v2.data)

    def save_weights(self):
        """保存模型"""
        torch.save(self.model.state_dict(), 'dqn_model.pth')


if __name__ == '__main__':
    model = ConvNet(input_shape=(10, 400, 1), output_shape=31)
    x = torch.randn(31, 1, 10, 400)  # Conv2d 要求的输入形状为 [batch_size, channels, height, width]
    y = model(x)
    print(y)
