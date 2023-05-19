"""
DPE 算法实现：
局限性：该算法的改进在于使用了倒数来分配流量，但是时间复杂度是极高的，只适合在简单功能下使用。
"""
from models.utils.dataset import *
from models.utils.text import SchedulingResult
from models.utils.tools import *

MAX_VALUE = 9e+4


class DPE:
    def __init__(self, cpus, bw, pp, simple_paths, reciprocal_list, proportion_list):
        # get the generated edge computing scenario
        self.cpus, self.bw, self.pp = cpus, bw, pp
        self.simple_paths, self.reciprocal_list, self.proportion_list = simple_paths, reciprocal_list, proportion_list

    def get_response_time(self, sorted_job_path=args.batch_task_topological_order_path):
        """
        Calculate the overall finish time of all DAGs achieved by DPE algorithm.
        """
        if not os.path.exists(sorted_job_path):
            print('DAGs\' topological order has not been obtained! Please get topological order firstly.')
            return

        df = pd.read_csv(sorted_job_path)  # 读取 batch_task_topological_order.csv
        rows = df.shape[0]
        idx = 0

        makespan_all = 0
        task_deployment_all = []
        cpu_earliest_finish_time_all = []
        task_start_time_all = []
        cpu_task_mapping_list_all = []

        total_job_nums = args.total_jobs  # 需要采样的 job 的数量
        calculated_num = 0  # 已经计算的 job，用于监控当前处理进度
        print('\nGetting makespan for %d cpus by DPE algorithm ...' % total_job_nums)

        bar = ProgressBar()
        count = 0

        # 遍历所有的 cpus
        while idx < rows:

            task_nums = 0

            # 对 jobs 中的每个 job 进行单独处理
            task_name = df.loc[idx, 'task_name'].split('_')[0]

            # if...else 一次处理一个 job
            if task_name == 'task' or task_name == "MergeTask":
                _, idx = get_one_job(df, idx)  # 向前推进 idx

            else:
                job, next_idx = get_one_job(df, idx)
                task_nums = next_idx - idx

                job_name = job.loc[idx, 'job_name']
                print(job_name)

                job_pp_required = np.load(args.task_depend_prefix + job_name + '_required_cpu.npy')
                job_data_stream = np.load(args.task_depend_prefix + job_name + '_data_size.npy')

                cpu_earliest_finish_time = np.zeros((task_nums, args.n_nodes))
                task_start_time = np.zeros(task_nums)  # 记录每个 task 的开始时间
                task_deployment = -1 * np.ones(task_nums)  # 记录每个 task 是否已被分配
                cpu_task_mapping_list = []  # 记录每个 cpu 上的 task 处理序列
                cpu_finish_time = np.zeros(args.n_nodes)  # 记录每个 cpu 上 `最近的 task 的完成时间`

                makespan = 0
                for j in range(task_nums + 1):  # +1 是因为人为创建了一个 dummy tail task
                    if j == task_nums:  # 处理 dummy tail task，用于连接所有的出口函数，更新 makespan
                        for e in range(task_nums):
                            if task_deployment[e] == -1.:
                                task_deployment[e] = int(
                                    np.argmin(cpu_earliest_finish_time[e]))  # 将任务分配到完成时间最早的 cpu 上
                                cpu_task_mapping_list.append(e + 1)  # e 是 task 编号，将 task 添加到 mapping list 中
                                if min(cpu_earliest_finish_time[e]) > makespan:
                                    makespan = min(cpu_earliest_finish_time[e])
                        break

                    # 获取 task 编号（int 类型）
                    task_name_list = job.loc[j + idx, 'task_name'].strip().split('_')
                    task_name_list_len = len(task_name_list)
                    task_name_len = len(task_name_list[0])
                    task_name = int(task_name_list[0][1:task_name_len])

                    if task_name_list_len == 1:  # 长度 == 1，说明这是个入口 task
                        pass
                    else:  # 根据 task 之间的依赖关系处理分配
                        for cpu_current in range(args.n_nodes):
                            # get t(p(f_j)) where p(f_j) is cpu_current
                            comp_cost = job_pp_required[task_name - 1] / self.pp[cpu_current]
                            all_min_phi = []  # 记录当前 cpu 上的 task 所依赖的所有其他 task 最早完成时间

                            # TODO: for 循环是 FIFO 模式，编号小的 task 优先被部署，这在实际场景下可能造成耗时过长，有待优化
                            # 如果同时有 M2, R4_2 和 R5_2, M2，我们怎么去选择先部署 M4 还是 R5？作者先部署 R4，因为编号 4 < 5。
                            for i in range(task_name_list_len - 1):
                                if task_name_list[i + 1] == '':
                                    continue
                                dependent_task_name = int(task_name_list[i + 1])

                                # Case 1: 若 dependent_task 之前已经被部署，那么直接获取 min_phi
                                if task_deployment[dependent_task_name - 1] != -1.:
                                    cpu_deployed_to = int(task_deployment[dependent_task_name - 1])
                                    if cpu_current == cpu_deployed_to:
                                        comm_cost = 0  # 若 dependent_task 与 curr_task 分配到同一个 cpu 上，通信成本为 0
                                    else:
                                        comm_cost = self.proportion_list[cpu_deployed_to][cpu_current] * \
                                                    job_data_stream[dependent_task_name - 1] * \
                                                    self.reciprocal_list[cpu_deployed_to][cpu_current][0]
                                    min_phi = cpu_earliest_finish_time[dependent_task_name - 1][
                                                  cpu_deployed_to] + comm_cost + comp_cost
                                    all_min_phi.append(min_phi)
                                    continue

                                # Case 2: 如果 dependent_task 没有部署过，先检查 dependent_task 所依赖 task 是否已经部署
                                for h in range(task_nums):
                                    task_name_list_of_dependent_task_depends = job.loc[
                                        h + idx, 'task_name'].strip().split('_')
                                    task_name_len_of_dependent_task_depends = len(
                                        task_name_list_of_dependent_task_depends[0])
                                    task_name_of_dependent_task_depends = task_name_list_of_dependent_task_depends[0][
                                                                          1:task_name_len_of_dependent_task_depends]

                                    # 用 if-else 找到 dependent_task 依赖的 task
                                    if int(task_name_of_dependent_task_depends) != dependent_task_name:
                                        continue

                                    else:  # 找到了 dependent_task_name
                                        name_str_list_inner_len = len(task_name_list_of_dependent_task_depends)

                                        if name_str_list_inner_len == 1:  # 入口函数直接设置它的 cpu_earliest_finish_time
                                            cpu_earliest_finish_time[dependent_task_name - 1] = \
                                                job_pp_required[dependent_task_name - 1] / self.pp + cpu_finish_time

                                        # 虽然 dependent_task_name 的 cpu_earliest_finish_time 已经被设置了，
                                        # cpu_finish_time 仍有可能被更新
                                        else:
                                            cpu_begin_time = np.zeros(args.n_nodes)

                                            for cpu_k in range(args.n_nodes):
                                                min_cpu_begin_time = 0

                                                # dependent_task_name 被部署到 cpu_k 上
                                                for h_inner in range(name_str_list_inner_len - 1):

                                                    # dependent_task_name 的一个 “前辈” 和它的部署情况
                                                    if task_name_list_of_dependent_task_depends[h_inner + 1] == '':
                                                        continue

                                                    dependent_predecessor_task_num = int(
                                                        task_name_list_of_dependent_task_depends[h_inner + 1])
                                                    predecessor_task_deployed_to = int(
                                                        task_deployment[dependent_predecessor_task_num - 1])

                                                    if predecessor_task_deployed_to == -1.:
                                                        print('Sth. wrong! It\'s impossible!')

                                                    if cpu_k == predecessor_task_deployed_to:
                                                        comm_cost = 0
                                                    else:
                                                        comm_cost = self.proportion_list[predecessor_task_deployed_to][
                                                                        cpu_k] * job_data_stream[
                                                                        dependent_task_name - 1] * \
                                                                    self.reciprocal_list[predecessor_task_deployed_to][
                                                                        cpu_k][0]
                                                    tmp = cpu_earliest_finish_time[dependent_predecessor_task_num - 1][
                                                              predecessor_task_deployed_to] + comm_cost

                                                    # 只有 dependent_task_name 最慢的 “前辈” 完成数据传输，dependent_task_name 才能开始执行
                                                    if tmp > min_cpu_begin_time:
                                                        min_cpu_begin_time = tmp

                                                if min_cpu_begin_time > cpu_finish_time[cpu_k]:
                                                    cpu_begin_time[cpu_k] = min_cpu_begin_time
                                                else:
                                                    cpu_begin_time[cpu_k] = cpu_finish_time[cpu_k]

                                            cpu_earliest_finish_time[dependent_task_name - 1] = \
                                                job_pp_required[dependent_task_name - 1] / self.pp + cpu_begin_time
                                        break

                                # 确定 dependent_task_name 的最佳部署 cpu
                                min_phi = MAX_VALUE
                                cpu_selected = -1

                                # 以遍历的方式确定最佳部署 cpu（有待优化）
                                for cpu_m in range(args.n_nodes):
                                    if cpu_current == cpu_m:
                                        comm_cost = 0
                                    else:
                                        comm_cost = self.proportion_list[cpu_m][cpu_current] * job_data_stream[
                                            dependent_task_name - 1] * self.reciprocal_list[cpu_m][cpu_current][0]
                                    phi = cpu_earliest_finish_time[dependent_task_name - 1][
                                              cpu_m] + comm_cost + comp_cost

                                    if phi < min_phi:
                                        min_phi = phi
                                        cpu_selected = cpu_m

                                # 整理 dependent_task_name 的部署结果
                                task_deployment[dependent_task_name - 1] = cpu_selected
                                cpu_task_mapping_list.append(dependent_task_name)
                                cpu_finish_time[cpu_selected] = \
                                    cpu_earliest_finish_time[dependent_task_name - 1][cpu_selected]
                                task_start_time[dependent_task_name - 1] = \
                                    cpu_finish_time[cpu_selected] - job_pp_required[dependent_task_name - 1] / self.pp[
                                        cpu_selected]
                                all_min_phi.append(min_phi)

                            # task 依赖所有 task 被部署成功，使用 max(all_min_phi) 更新 task 的 cpu_earliest_finish_time
                            cpu_earliest_finish_time[task_name - 1][cpu_current] = max(all_min_phi)

                makespan_all += makespan  # 累计所有 job 的 makespan
                task_deployment_all.append(task_deployment)  # 每部署完一个 job，将其 append 到 task_deployment_all 中
                cpu_task_mapping_list_all.append(cpu_task_mapping_list)
                cpu_earliest_finish_time_all.append(cpu_earliest_finish_time)  # 记录所有 job 的最早完成时间
                task_start_time_all.append(task_start_time)  # 记录所有所有 job 的最早开始时间
                count += 1

                print(makespan)

            calculated_num += 1
            percent = calculated_num / float(total_job_nums) * 100
            # for overflow
            if percent > 100:
                percent = 100
            bar.update(percent)
            idx += task_nums

        # 打印调度结果
        for i in range(count):
            scheduling_result = SchedulingResult(cpu_earliest_finish_time_all,
                                                 task_deployment_all,
                                                 cpu_task_mapping_list_all,
                                                 task_start_time_all,
                                                 i)
            scheduling_result.print()

        print('The overall makespan achieved by DPE: %f seconds' % makespan_all)
        print('The average makespan: %f seconds' % (makespan_all / total_job_nums))
        makespan_avg = makespan_all / total_job_nums * 1.0

        return cpu_earliest_finish_time_all, task_deployment_all, cpu_task_mapping_list_all, task_start_time_all, makespan_avg
