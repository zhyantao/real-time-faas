import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from param import *
from multi_resource_env.env import MultiResEnvironment as Env
from multi_resource_agents.actor_agent import MultiResActorAgent
from compute_gradients import *


def invoke_model(actor_agent, obs, exp):
    # parse observation
    job_dags, source_job, num_source_exec, \
        frontier_nodes, exec_commit, \
        moving_executors, action_map = obs

    if sum([len(frontier_nodes[n]) \
            for n in frontier_nodes]) == 0:
        # no action to take
        exec_idx = next(x[0] for x in \
                        enumerate(num_source_exec) if x[1] > 0)
        return None, exec_idx, num_source_exec[exec_idx]

    # invoking the learning model
    node_act, job_act, \
        node_act_probs, job_act_probs, \
        node_inputs, job_inputs, \
        node_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, \
        exec_map, job_dags_changed = \
        actor_agent.invoke_model(obs)

    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        exec_idx = next(x[0] for x in \
                        enumerate(num_source_exec) if x[1] > 0)
        return None, exec_idx, num_source_exec[exec_idx]

    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = action_map[int(np.floor(node_act[0] / len(num_source_exec)))]
    use_exec_type = node_act[0] % len(num_source_exec)

    # node should be valid in the frontier nodes
    assert node in frontier_nodes[use_exec_type]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
                             len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - \
                         exec_map[node.job_dag] + \
                         num_source_exec[use_exec_type]
    else:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec[use_exec_type])

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

    return node, use_exec_type, use_exec


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # model evaluation seed
    tf.set_random_seed(agent_id)

    # set up environment
    env = Env()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = MultiResActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, sum(args.exec_group_num) + 1), args.exec_mems)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, max_time, entropy_weight) = \
            param_queue.get()

        # synchronize model
        actor_agent.set_params(actor_params)

        # try one round of experimets
        try:
            # reset environment
            env.seed(seed)
            env.reset(max_time=max_time)

            # set up storage for experience
            exp = {'node_inputs': [], 'job_inputs': [],
                   'gcn_mats': [], 'gcn_masks': [],
                   'summ_mats': [], 'running_dag_mat': [],
                   'dag_summ_back_mat': [],
                   'node_act_vec': [], 'job_act_vec': [],
                   'node_valid_mask': [], 'job_valid_mask': [],
                   'reward': [], 'wall_time': [],
                   'job_state_change': []}

            # run experiment
            obs = env.observe()
            done = False

            # initial time
            exp['wall_time'].append(env.wall_time.curr_time)

            while not done:

                node, use_exec_type, use_exec = \
                    invoke_model(actor_agent, obs, exp)

                obs, reward, done = \
                    env.step(node, use_exec_type, use_exec)

                if node is not None:
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)
                elif len(exp['reward']) > 0:
                    # Note: if we skip the reward when node is None
                    # (i.e., no available actions), the sneaky
                    # agent will learn to exhaustively pick all
                    # nodes in one scheduling round, in order to
                    # avoid the negative reward
                    exp['reward'][-1] += reward
                    exp['wall_time'][-1] = env.wall_time.curr_time

            # report reward signals to master
            assert len(exp['node_inputs']) == len(exp['reward'])
            reward_queue.put(
                [exp['reward'], exp['wall_time'],
                 len(env.finished_job_dags),
                 np.mean([j.completion_time - j.start_time \
                          for j in env.finished_job_dags]),
                 env.wall_time.curr_time >= env.max_time])

        # environment interaction catch
        except:
            reward_queue.put(None)

        # get advantage term from master
        batch_adv = adv_queue.get()

        # compute gradients
        if batch_adv is not None:
            try:
                actor_gradient, loss = compute_actor_gradients(
                    actor_agent, exp, batch_adv, entropy_weight)

                # report gradient to master
                gradient_queue.put([actor_gradient, loss])
            except:
                gradient_queue.put(None)
