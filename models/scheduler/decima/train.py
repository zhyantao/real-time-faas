import numpy as np
import tensorflow as tf

from actor_agent import ActorAgent
from compute_gradients import compute_actor_gradients
from envs import Environment
from param import args


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed
    tf.set_random_seed(agent_id)

    # set up environment
    env = Environment()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, max_time, entropy_weight) = param_queue.get()

        # synchronize model
        actor_agent.set_params(actor_params)

        # reset environment
        env.seed(seed)  # 环境重置
        env.reset(max_time=max_time)

        # set up storage for experience
        exp = {
            'node_inputs': [], 'job_inputs': [], 'gcn_mats': [], 'gcn_masks': [],
            'summ_mats': [], 'running_dag_mat': [], 'dag_summ_back_mat': [],
            'node_act_vec': [], 'job_act_vec': [], 'node_valid_mask': [], 'job_valid_mask': [],
            'reward': [], 'wall_time': [], 'job_state_change': []
        }  # 用于存储强化学习过程中的经验

        try:
            # The masking functions (node_valid_mask and job_valid_mask in actor_agent.py) has some
            # small chance (once in every few thousand iterations) to leave some non-zero probability
            # mass for a masked-out action.
            # This will trigger the check in "node_act and job_act should be valid" in actor_agent.py.
            # Whenever this is detected, we throw out the rollout of that iteration and try again.

            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)  # wall_time 表示真实的当前时间

            while not done:

                node, use_exec = invoke_model(actor_agent, obs, exp)  # 开始强化学习

                obs, reward, done = env.step(node, use_exec)  # 根据决策获得奖励，并观察是否已经结束

                if node is not None:  # 仅在存储中追加有效值
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)

                elif len(exp['reward']) > 0:  # 如果当前决策下没有可分配的 node，修改 reward 中的最后一个值
                    # Note: if we skip the reward when node is None (i.e., no available actions),
                    # the sneaky agent will learn to exhaustively pick all nodes in one scheduling round,
                    # in order to avoid the negative reward
                    exp['reward'][-1] += reward
                    exp['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            assert len(exp['node_inputs']) == len(exp['reward'])  # 所有的输入节点都被分配到了任务上
            reward_queue.put([
                exp['reward'], exp['wall_time'],
                len(env.finished_job_dags),
                np.mean([j.completion_time - j.start_time for j in env.finished_job_dags]),
                env.wall_time.curr_time >= env.max_time])

            # get advantage term from master
            batch_adv = adv_queue.get()  # 啥叫从 master 上获取有利条件？

            if batch_adv is None:  #
                # some other agents panic for the try and the main thread throw out the rollout,
                # reset and try again now
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(actor_agent, exp, batch_adv, entropy_weight)

            # report gradient to master
            gradient_queue.put([actor_gradient, loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # need to still get from adv_queue to
            # prevent blocking
            adv_queue.get()


def invoke_model(actor_agent, obs, exp):
    # parse observation
    # 获取环境中观察到的值
    job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, moving_executors, action_map \
        = obs

    if len(frontier_nodes) == 0:
        # no action to take
        return None, num_source_exec

    # invoking the learning model
    node_act, job_act, node_act_probs, job_act_probs, node_inputs, job_inputs, node_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map, exec_map, job_dags_changed \
        = actor_agent.invoke_model(obs)  # 调用模型，开始分配

    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, num_source_exec

    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = action_map[node_act[0]]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag] + num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(node.num_tasks - node.next_task_idx - exec_commit.node_commit[node] - moving_executors.count(node),
                   agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec
