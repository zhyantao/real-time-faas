import tensorflow as tf
import tensorflow.contrib.layers as tl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from gcn import GraphCNN
from gsn import GraphSNN
from param import args
from tf_op import expand_act_on_state


def leaky_relu(features, alpha=0.2, name=None):
    """Compute the Leaky ReLU activation function.
    "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
    AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
    http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    Args:
      features: A `Tensor` representing preactivation values.
      alpha: Slope of the activation function at x < 0.
      name: A name for the operation (optional).
    Returns:
      The activation value.
    """
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


class Agent(object):
    # super class of scheduling agent

    def __init__(self):
        pass

    def get_action(self, obs):
        print('get_action not implemented')
        exit(1)


class ActorAgent(Agent):
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims, output_dim, max_depth, executor_levels,
                 eps=1e-6, act_fn=leaky_relu, optimizer=tf.train.AdamOptimizer, scope='actor_agent'):
        """

        :param sess:
        :param node_input_dim: node input dimensions to graph embedding (default: 5)
        :param job_input_dim: job input dimensions to graph embedding (default: 3)
        :param hid_dims: hidden dimensions throughout graph embedding (default: [16, 8])
        :param output_dim: output dimensions throughout graph embedding (default: 8)
        :param max_depth: Maximum depth of root-leaf message passing (default: 8)
        :param executor_levels: TODO (default: [1, 100])
        :param eps: epsilon (default: 1e-6)
        :param act_fn: 激活函数 leaky_relu
        :param optimizer: 优化器 Adam
        :param scope: (default: actor_agent)
        """

        Agent.__init__(self)

        self.sess = sess
        self.node_input_dim = node_input_dim  # job 是含有依赖的 task 的集合，node 是具有连通性的 machine 的集合
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # node input dimension: [total_num_nodes, num_features]
        # total_num_nodes 任意，但是 features 是固定的
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])

        # job input dimension: [total_num_jobs, num_features]
        # total_num_jobs 任意，但是 features 是固定的
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])

        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),  # 输入数据：其中一部分是 DAG 的嵌入信息，另一个部分是 node 的拓扑连接
            self.node_input_dim + self.output_dim,
            self.hid_dims,  # GraphSNN 的输入维度：self.node_input_dim + self.output_dim
            self.output_dim, self.act_fn, self.scope)

        # valid mask for node action ([batch_size, total_num_nodes])
        # node valid mask 用于检查 node 是否有效，有效则为 1
        self.node_valid_mask = tf.placeholder(tf.float32, [None, None])

        # valid mask for executor limit on jobs ([batch_size, num_jobs * num_exec_limits])
        # job valid mask 用于检查 job 是否有效，有效则为 1
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])

        # map back the dag summarization to each node ([total_num_nodes, num_dags])
        # TODO: 为什么需要 map back
        self.dag_summ_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        self.node_act_probs, self.job_act_probs = self.actor_network(
            self.node_inputs, self.gcn.outputs, self.job_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.node_valid_mask, self.job_valid_mask,
            self.dag_summ_backward_map, self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = tf.log(self.node_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.node_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 1)

        # job_acts [batch_size, num_jobs, 1]
        logits = tf.log(self.job_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Carlo or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs, self.node_act_vec),
            reduction_indices=1, keep_dims=True)

        # select job action probability
        self.selected_job_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.job_act_probs, self.job_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # actor loss due to advantage (negated)
        self.adv_loss = tf.reduce_sum(
            tf.multiply(
                tf.log(self.selected_node_prob * self.selected_job_prob + self.eps),
                -self.adv))

        # node_entropy
        self.node_entropy = tf.reduce_sum(
            tf.multiply(
                self.node_act_probs,
                tf.log(self.node_act_probs + self.eps)))

        # prob on each job
        self.prob_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0], tf.reshape(self.node_act_probs, [-1, 1])),
            [tf.shape(self.node_act_probs)[0], -1])

        # job entropy
        self.job_entropy = tf.reduce_sum(
            tf.multiply(self.prob_each_job,
                        tf.reduce_sum(tf.multiply(
                            self.job_act_probs,
                            tf.log(self.job_act_probs + self.eps)),
                            reduction_indices=2)))

        # entropy loss
        self.entropy_loss = self.node_entropy + self.job_entropy

        # normalize entropy
        self.entropy_loss /= (tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)) +
                              tf.log(float(len(self.executor_levels))))
        # normalize over batch size (note: adv_loss is sum)
        # * tf.cast(tf.shape(self.node_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).apply_gradients(zip(self.act_gradients, self.params))

        # network parameter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, node_inputs, gcn_outputs, job_inputs, gsn_dag_summary, gsn_global_summary,
                      node_valid_mask, job_valid_mask, gsn_summ_backward_map, act_fn):
        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_valid_mask)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(node_inputs, [batch_size, -1, self.node_input_dim])

        # (2) reshape job inputs to batch format
        job_inputs_reshape = tf.reshape(job_inputs, [batch_size, -1, self.job_input_dim])

        # (3) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(gcn_outputs, [batch_size, -1, self.output_dim])

        # (4) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = tf.reshape(gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_dag_summ_extend = tf.matmul(gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (5) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = tf.reshape(gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile(gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        gsn_global_summ_extend_node = tf.tile(gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend)[1], 1])

        # (6) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_node = tf.concat([
                node_inputs_reshape,
                gcn_outputs_reshape,
                gsn_dag_summ_extend,
                gsn_global_summ_extend_node],
                axis=2)

            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_nodes)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # valid mask on node
            node_valid_mask = (node_valid_mask - 1) * 10000.0

            # apply mask
            node_outputs = node_outputs + node_valid_mask

            # do mask softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)

            # -- part B, the distribution over executor limits --
            merge_job = tf.concat([
                job_inputs_reshape,
                gsn_dag_summ_reshape,
                gsn_global_summ_extend_job],
                axis=2)

            expanded_state = expand_act_on_state(merge_job, [l / 50.0 for l in self.executor_levels])

            job_hid_0 = tl.fully_connected(expanded_state, 32, activation_fn=act_fn)
            job_hid_1 = tl.fully_connected(job_hid_0, 16, activation_fn=act_fn)
            job_hid_2 = tl.fully_connected(job_hid_1, 8, activation_fn=act_fn)
            job_outputs = tl.fully_connected(job_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])

            # valid mask on job
            job_valid_mask = (job_valid_mask - 1) * 10000.0

            # apply mask  # valid mask 是什么东西？
            job_outputs = job_outputs + job_valid_mask

            # reshape output dimension for softmax the executor limits
            # (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1, len(self.executor_levels)])

            # do mask softmax over jobs
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            return node_outputs, job_outputs

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def predict(self, node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks,
                summ_mats, running_dags_mat, dag_summ_backward_map):
        return self.sess.run(
            [  # 第一个参数用于保存计算结果，在本例中，将计算结果保存到 self.node_act_probs 等四个参数上
                self.node_act_probs,
                self.job_act_probs,
                self.node_acts,
                self.job_acts
            ],
            feed_dict={  # feed_dict 用作占位符，每次神经网络训练时使用的数据
                i: d for i, d in zip(
                    # key:
                    [self.node_inputs]  # (1) Tensor("Placeholder_0", shape=(?, 5), dtype=float32)
                    + [self.job_inputs]  # (2) Tensor("Placeholder_1:0", shape=(?, 3), dtype=float32)
                    + [self.node_valid_mask]  # (3) Tensor("Placeholder_40:0", shape=(?, ?), dtype=float32)
                    + [self.job_valid_mask]  # (4) Tensor("Placeholder_41:0", shape=(?, ?), dtype=float32)
                    + self.gcn.adj_mats  # (5) Tensor("Placeholder_4:0", shape=(?, 2), dtype=int64)
                    + self.gcn.masks  # (6) Tensor("Placeholder_26:0", shape=(?, 1), dtype=int64)
                    + self.gsn.summ_mats  # (7) Tensor("Placeholder_27:0", shape=(?, 1), dtype=int64)
                    + [self.dag_summ_backward_map],  # (8) Tensor("Placeholder_42:0", shape=(?, ?), dtype=int64)
                    # value:
                    [node_inputs]  # ndarray: (80, 5)
                    + [job_inputs]  # ndarray: (10, 3)
                    + [node_valid_mask]  # ndarray: (1, 80)
                    + [job_valid_mask]  # ndarray: (1, 1000)
                    + gcn_mats  # ndarray: (80, 1)
                    + gcn_masks  # ndarray: (80, 80)
                    + [summ_mats, running_dags_mat]  # ndarray: (80, 80)
                    + [dag_summ_backward_map]  # ndarray: (80, 10)
                )
            }
        )

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
                             feed_dict={i: d for i, d in zip(
                                 [self.node_inputs] + self.gsn.summ_mats,
                                 [node_inputs] + summ_mats)
                                        })

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats,
                      running_dags_mat, dag_summ_backward_map, node_act_vec, job_act_vec, adv, entropy_weight):

        return self.sess.run(
            [self.act_gradients, [self.adv_loss, self.entropy_loss]],

            feed_dict={i: d for i, d in zip(
                [self.node_inputs] +
                [self.job_inputs] +
                [self.node_valid_mask] +
                [self.job_valid_mask] +
                self.gcn.adj_mats +
                self.gcn.masks +
                self.gsn.summ_mats +
                [self.dag_summ_backward_map] +
                [self.node_act_vec] +
                [self.job_act_vec] +
                [self.adv] +
                [self.entropy_weight],

                [node_inputs] +
                [job_inputs] +
                [node_valid_mask] +
                [job_valid_mask] +
                gcn_mats + gcn_masks +
                [summ_mats, running_dags_mat] +
                [dag_summ_backward_map] +
                [node_act_vec] +
                [job_act_vec] +
                [adv] +
                [entropy_weight]
            )})
