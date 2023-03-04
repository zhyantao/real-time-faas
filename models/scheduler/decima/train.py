import multiprocessing as mp
import time

from average_reward import *
from compute_baselines import *
from tf_logger import TFLogger
from train_agent import *

# os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 去掉 warning 信息，只显示 error


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(
        sess, args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1))

    # tensorboard logging
    tf_logger = TFLogger(sess, [
        'actor_loss', 'entropy', 'value_loss', 'episode_length',
        'average_reward_per_second', 'sum_reward', 'reset_probability',
        'num_jobs', 'reset_hit', 'average_job_duration',
        'entropy_weight'])

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([
                actor_params, args.seed + ep,
                max_time, entropy_weight])

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
            all_num_finished_jobs, all_avg_job_duration, \
            all_reset_hit, = [], [], [], [], [], []

        t1 = time.time()

        # get reward from agents
        any_agent_panic = False

        for i in range(args.num_agents):
            result = reward_queues[i].get()

            if result is None:
                any_agent_panic = True
                continue
            else:
                batch_reward, batch_time, \
                    num_finished_jobs, avg_job_duration, \
                    reset_hit = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues[i].put(None)
            continue

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean([cr[0] for cr in all_cum_reward]),
            reset_prob,
            np.mean(all_num_finished_jobs),
            np.mean(all_reset_hit),
            np.mean(all_avg_job_duration),
            entropy_weight])

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
                                      args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
                                  args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                                   'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()
