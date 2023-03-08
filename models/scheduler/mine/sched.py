import datetime
import json
import os
from abc import ABC, abstractmethod

import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dqn import DQN
from envs import Environment
from node import Node
from utils import _load_tasks


class Action(object):
    """
    调度行为
    """

    def __init__(self, task, node):
        self.task = task
        self.node = node

    def __repr__(self):
        return 'Action(task={0} -> node={1})'.format(self.task.label, self.node.label)


class Scheduler(ABC):
    """
    接口函数：调度器
    """

    @abstractmethod
    def schedule(self):
        pass


class DeepRMScheduler(Scheduler):
    """DeepRM scheduler."""

    def __init__(self, environment, train=True):
        if train:
            DeepRMTrainer(environment)._train()
        # print('sched.py --> environment.summary(): ')
        # print(environment.summary())
        # print('sched.py --> environment.summary() end')
        input_shape = environment.summary().shape  # 输入形状 (H, W)
        output_shape = environment.queue_size * len(environment.nodes) + 1  # 输出是队列中每个元素分配到 node 上
        # print('sched.py --> input_shape, output_shape: ')
        # print(input_shape, output_shape)
        # print('sched.py --> input_shape, output_shape end')
        self.dqn_train = DQN(input_shape, output_shape)
        self.environment = environment

    def schedule(self):
        """Schedule with trained model."""
        actions = []

        # apply actions until there's an invalid one
        while True:
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, 0)
            task_index, node_index = self._explain(action_index)
            if task_index < 0 or node_index < 0:
                break
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                break
            del self.environment.queue[task_index]
            actions.append(Action(scheduled_task, scheduled_node))

        # proceed to the next timestep
        self.environment.timestep()

        return actions

    def _explain(self, action_index):
        """Explain action."""
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit * node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return task_index, node_index


class DeepRMTrainer(nn.Module):
    """DeepRM Trainer."""

    def __init__(self, environment):
        super().__init__()
        self.episodes = 100
        self.copy_steps = 32
        self.save_steps = 32
        self.epsilon = 0.99
        self.decay = 0.99
        self.min_epsilon = 0.1
        input_shape = environment.summary().shape  # (H, W)
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.dqn_target = DQN(input_shape, output_shape)
        self.total_rewards = np.empty(self.episodes)
        self.environment = environment
        if not os.path.exists('__cache__/summary'):
            os.makedirs('__cache__/summary')
        self.summary_writer = SummaryWriter(
            '__cache__/summary/dqn-{0}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def _train(self):
        """
        训练过程
        :return:
        """
        for i in range(self.episodes):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            self.total_rewards[i] = self.train_episode()
            # job slowdown is the negative total reward
            self.summary_writer.add_scalar('Episode Job Slowdown', -self.total_rewards[i])
            print('Episode {0} Job Slowdown {1}'.format(i, -self.total_rewards[i]))

    def train_episode(self):
        """
        每个 train episode 需要做的事情
        :return:
        """
        rewards = 0
        step = 0
        self.environment, _ = load(load_scheduler=False)
        while not self.environment.terminated():
            # observe state and predict action
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, self.epsilon)
            task_index, node_index = self._explain(action_index)

            # invalid action, proceed to the next timestep
            if task_index < 0 or node_index < 0:
                self.environment.timestep()
                continue
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                self.environment.timestep()
                continue

            # apply action, calculate reward and train model
            del self.environment.queue[task_index]
            prev_observation = observation
            reward = self.environment.reward()
            observation = self.environment.summary()
            rewards = rewards + reward
            exp = {'s': prev_observation, 'a': action_index, 'r': reward, 's2': observation,
                   'done': self.environment.terminated()}
            self.dqn_train.add_experience(exp)
            self.dqn_train._train(self.dqn_target)

            step += 1
            # copy weights from train model to target model periodically
            if step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)
            # save model periodically
            if step % self.save_steps == 0:
                self.dqn_target.save_weights()

        return rewards

    def _explain(self, action_index):
        """
        解释动作
        :param action_index:
        :return:
        """
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit * node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return task_index, node_index


def load(load_environment=True, load_scheduler=True):
    """
    从 conf/env.conf.json 中加载环境和调度器
    :param load_environment:
    :param load_scheduler:
    :return:
    """
    tasks = _load_tasks()  # [ Task(resources=[2, 3, 3], duration=2, label=task1), ... ]
    task_generator = (t for t in tasks)

    with open('../../../configs/env.conf.json', 'r') as fr:
        data = json.load(fr)

        # 获取 node 信息
        nodes = []  # 共三个 node
        label = 0
        for node_json in data['nodes']:
            label += 1
            nodes.append(Node(node_json['resource_capacity'],
                              node_json['duration_capacity'],
                              'node' + str(label)))

        # 生成状态空间
        # 从这里生成状态空间，Environment( nodes, queue, backlog, tasks)
        # 可以观察到状态空间共分成四部分，分别代表了运行节点、工作队列、积压队列、任务
        environment = None
        if load_environment:
            environment = Environment(nodes, data['queue_size'], data['backlog_size'], task_generator)
            # print('environment: ' + environment)  # 打印状态空间
            environment.timestep()  # 前进到下一个时间步

        # 生成动作空间
        scheduler = None
        if load_scheduler:
            scheduler = DeepRMScheduler(environment, data['train'])

        return environment, scheduler


if __name__ == '__main__':
    environment, scheduler = load()
    while not environment.terminated():
        environment.plot()
        actions = scheduler.schedule()
        # print(actions)
    print('END')
