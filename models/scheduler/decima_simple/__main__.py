import numpy as np
import tensorflow as tf

from network import ActorAgent
from param import args

if __name__ == '__main__':
    node_inputs = np.random.random((80, 5))
    job_inputs = np.random.random((10, 3))
    node_valid_mask = np.random.random((1, 80))
    job_valid_mask = np.random.random((1, 1000))
    gcn_mats = list([np.random.random((80, 1)) for _ in range(8)])
    gcn_masks = list([np.random.random((80, 80)) for _ in range(8)])
    summ_mats = np.random.random((10, 80))
    running_dags_mat = np.random.random((1, 10))
    dag_summ_backward_map = np.random.random((80, 10))

    sess = tf.Session()

    actor_agent = ActorAgent(  # 初始化智能体
        sess, args.node_input_dim, args.job_input_dim,  # sess 的作用：存储用户指定的 tf 环境
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    actor_agent.predict(node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks,
                        summ_mats, running_dags_mat, dag_summ_backward_map)

    sess.close()
