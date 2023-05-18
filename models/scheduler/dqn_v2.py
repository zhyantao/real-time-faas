"""Take from https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95"""
import collections
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm


class ReplayBuffer:
    """经验回放池：用于存放最近的 transitions"""

    def __init__(self, capacity):
        """队列实现"""
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """将 transitions 加入 buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从 buffer 中采样 batch_size 条数据"""
        transitions = random.sample(self.buffer, batch_size)  # 转换单元：从 (state, action) 映射到 (next_state, reward)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """返回 buffer 中的数据总量"""
        return len(self.buffer)


class QNet(nn.Module):
    """只有一层隐藏层的 Q 网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvQNet(nn.Module):
    """使用带卷积的 Q 网络"""

    def __init__(self, action_dim, in_channels=4):
        super(ConvQNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim

        # 初始化 Q 网络
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)

        # 初始化目标网络
        self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)

        # 初始化优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 记录更新次数
        self.device = device

    def take_action(self, state):
        """以 epsilon 的概率从经验池中选择奖励最大的动作，否则选择未知的动作"""

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()

        return action

    def update(self, transition_dict):
        """根据转移字典计算下一阶段的状态和动作"""
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大 Q 值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # 时序差分误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差
        self.optimizer.zero_grad()  # PyTorch 默认会累积梯度，这里对梯度手动清零
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:  # 每隔 target_update 次，更新目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


if __name__ == '__main__':
    learning_rate = 0.001
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.99
    epsilon = 0.01
    target_update = 10
    buffer_size = 8196
    minimal_size = 500
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make('CartPole-v1')

    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # actions 的数量
    agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
