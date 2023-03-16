from input_param import *
from network import ActorAgent
from param import args

if __name__ == '__main__':
    sess = tf.Session()

    # 状态空间
    node_inputs = get_node_inputs()
    job_inputs = get_job_inputs()
    node_valid_mask = get_node_valid_mask()
    job_valid_mask = get_job_valid_mask()
    gcn_mats = [item.eval(session=sess) for item in get_gcn_mats()]
    gcn_masks = get_gcn_masks()
    summ_mats = get_sum_mats().eval(session=sess)
    running_dags_mat = get_running_dags_mat().eval(session=sess)
    dag_summ_backward_map = get_dag_summ_backward_map()

    # 初始化状态空间
    actor_agent = ActorAgent(  # 初始化智能体
        sess, args.node_input_dim, args.job_input_dim,  # sess 的作用：存储用户指定的 tf 环境
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # 训练模型
    # set up storage for experience
    # 用于存储强化学习过程中的经验  【这些是在训练过程中需要学习的参数】
    exp = {'node_inputs': [], 'job_inputs': [], 'gcn_mats': [], 'gcn_masks': [],
           'summ_mats': [], 'running_dag_mat': [], 'dag_summ_back_mat': [],
           'node_act_vec': [], 'job_act_vec': [], 'node_valid_mask': [], 'job_valid_mask': [],
           'reward': [], 'wall_time': [], 'job_state_change': []}

    # 模型训练
    done = False
    while not done:

        node_act_probs, job_act_probs, node_acts, job_acts \
            = actor_agent.predict(node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks,
                                  summ_mats, running_dags_mat, dag_summ_backward_map)

        node_number = node_acts[0]  # 找到的节点编号
        assert node_valid_mask[0, node_number] == 1  # 确保分配有效

        job_idx = 6  # 在 job_dag 中找到这个 node 对应的 job id，现在设置的是一个假的值
        assert job_valid_mask[0, job_acts[0, job_idx] + len(actor_agent.executor_levels) * job_idx] == 1  # 确定该 job 有效

        # agent_exec_act = actor_agent.executor_levels[job_acts[0, job_idx]] - exec_map[node.job_dag]
        agent_exec_act = actor_agent.executor_levels[job_acts[0, job_idx]] - 0

        # use_exec = min(node.num_tasks - node.next_task_idx - exec_commit.node_commit[node] -
        # moving_executors.count(node), agent_exec_act, num_source_exec)
        use_exec = 4  # 最大并行数量限制

        # for storing the action vector in experience
        node_act_vec = np.zeros(node_act_probs.shape)
        node_act_vec[0, node_number] = 1

        # for storing job index
        job_act_vec = np.zeros(job_act_probs.shape)
        job_act_vec[0, job_idx, job_acts[0, job_idx]] = 1

        job_dags_changed = True

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

        # return node, use_exec
        reward = 0.0
        done = False

        # given an episode of experience (advantage computed from baseline)
        batch_size = node_act_vec.shape[0]

        entropy_weight = 1
        adv = np.array([[0], [0.345]])

        # compute gradient
        act_gradients, loss = actor_agent.get_gradients(
            node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks,
            summ_mats, running_dags_mat,
            dag_summ_backward_map, node_act_vec, job_act_vec,
            adv, entropy_weight)

        print('act_gradients: ', act_gradients)
        print('loss: ', loss)

    sess.close()
