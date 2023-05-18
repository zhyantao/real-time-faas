"""
Implement the HEFT algorithm.
This script refers the code from https://github.com/mrocklin/heft/.

Cast of Characters:

job - the job to be allocated
orders - dict {agent: [jobs-run-on-agent-in-order]}，可理解为 agent 上的 event 列表
jobson - dict {job: agent-on-which-job-is-run}，可理解为运行 job 的 agent 列表
prev - dict {job: (jobs which directly precede job)}，可理解为 job 的直接前驱列表
succ - dict {job: (jobs which directly succeed job)}，可理解为 job 的直接后继列表
comp_cost - function :: job, agent -> time to compute job on agent
comm_cost - function :: job, job, agent, agent -> time to transfer results
                        of one job needed by another between two agents

(function ---> job, edge server ---> agent)
"""
from collections import namedtuple
from functools import partial
from itertools import chain

from models.utils.dataset import *
from models.utils.tools import *


def get_agents():
    """
    默认设置了 4 台 server，编号从 0 开始。
    作者代码中的 server 和 agent 是同一个东西，都是能够运行代码的处理单元。
    function 和 task 是同一个东西，都代表 FaaS 中的一个函数实例。
    """
    servers = [str(n) for n in range(args.n_nodes)]
    return ''.join(servers)


all_agents = get_agents()
Event = namedtuple('Event', 'job start end')


class HEFT:
    def __init__(self, jobs, bw, pp, simple_paths, reciprocal_list, proportion_list):
        # get the generated edge computing scenario
        self.jobs, self.bw, self.pp = jobs, bw, pp
        self.simple_paths, self.reciprocal_list, self.proportion_list = simple_paths, reciprocal_list, proportion_list

    def get_response_time(self, sorted_job_path=args.batch_task_topological_order_path):
        """
        Calculate the overall finish time of all DAGs achieved by HEFT algorithm.
        """
        if not os.path.exists(sorted_job_path):
            print('Jobs\' topological order has not been obtained! Please get topological order firstly.')
            return

        df = pd.read_csv(sorted_job_path)
        rows = df.shape[0]
        idx = 0

        makespan_all = 0
        task_deployment_all = []
        cpu_task_mapping_list_all = []

        bar = ProgressBar()
        total_job_nums = args.total_jobs
        calculated_num = 0
        print('\nGetting makespan for %d jobs by HEFT algorithm ...' % total_job_nums)
        while idx < rows:

            # get a job
            task_nums = 0

            task_name = df.loc[idx, 'task_name'].split('_')[0]
            job_name = df.loc[idx, 'job_name']
            print(job_name)

            if task_name == 'task' or task_name == "MergeTask":
                _, idx = get_one_job(df, idx)  # 仅增加 idx

            else:
                job, next_idx = get_one_job(df, idx)
                task_nums = next_idx - idx

                # 设置 CPU 需求 和 数据大小
                job_pp_required = np.load(args.task_depend_prefix + job_name + '_required_cpu.npy')
                job_data_stream = np.load(args.task_depend_prefix + job_name + '_data_size.npy')

                # 获取 job 信息
                task_name_list = HEFT.get_task_name_list(job, idx, task_nums)
                succ = HEFT.get_succ(job, idx, task_nums)  # dict 类型
                comp_cost_array = self.get_comp_cost(task_name_list, job_pp_required)  # 获取计算成本
                comm_cost_array = self.get_comm_cost(succ, job_data_stream)  # 获取通信成本

                # 根据计算成本和通信成本将 task 分散到 server 上
                orders, jobson, makespan = HEFT.schedule(succ, all_agents,
                                                         HEFT.comp_cost, comp_cost_array,
                                                         HEFT.comm_cost, comm_cost_array)

                makespan_all += makespan
                task_deployment_all.append(jobson)
                cpu_task_mapping_list_all.append(orders)

                print(makespan)

            calculated_num += 1
            percent = calculated_num / float(total_job_nums) * 100
            # for overflow
            if percent > 100:
                percent = 100
            bar.update(percent)
            idx += task_nums

        # print('----------> cpu task mapping list all: \n')
        # pprint.pprint(cpu_task_mapping_list_all)
        # print('----------> task deployment all: \n')
        # pprint.pprint(task_deployment_all)

        print('The overall makespan achieved by HEFT: %f second' % makespan_all)
        print('The average makespan: %f second' % (makespan_all / total_job_nums))
        makespan_avg = makespan_all / total_job_nums * 1.0
        return cpu_task_mapping_list_all, task_deployment_all, makespan_avg

    @staticmethod
    def get_task_name_list(job, idx, task_nums):
        """
        根据拓扑排序后的结果，获取给定 job 中所有的 task 编号
        """
        task_num_list = []
        for i in range(task_nums):
            name_str_list = job.loc[i + idx, 'task_name'].strip().split('_')
            task_name_str_len = len(name_str_list[0])
            task_num = int(name_str_list[0][1:task_name_str_len])
            task_num_list.append(task_num)
        return task_num_list

    @staticmethod
    def get_succ(DAG, idx, DAG_len):
        """
        获取 DAG 中所有结点的直接后继.

        Get a DAG structure from the dataset. For example: for DAG
            "M1,12846.0,j_3,1,Terminated,157213,157295,100.0,0.3
            R2_1,371.0,j_3,1,Terminated,157297,157322,100.0,0.49
            R3,371.0,j_3,1,Terminated,157297,157325,100.0,0.49
            M4,1.0,j_3,1,Terminated,157322,157328,100.0,0.39
            R5,1.0,j_3,1,Terminated,157326,157330,100.0,0.39
            M6,1.0,j_3,1,Terminated,157326,157330,100.0,0.39
            M7,1.0,j_3,1,Terminated,157326,157330,100.0,0.39
            J8_6_7,1111.0,j_3,1,Terminated,157329,157376,100.0,0.59
            R9,1.0,j_3,1,Terminated,157376,157381,100.0,0.39
            J10_8_9,1111.0,j_3,1,Terminated,157331,157376,100.0,0.59
            R11_5_10,1.0,j_3,1,Terminated,157376,157381,100.0,0.39
            R12_4_11,1.0,j_3,1,Terminated,157376,157381,100.0,0.39
            R13_2_3_12,1.0,j_3,1,Terminated,157376,157381,100.0,0.39
            R14_13,1.0,j_3,1,Terminated,157376,157381,100.0,0.39",
        the output is
            {1: (2,),
             2: (13,),
             3: (13,),
             4: (12,),
             5: (11,),
             6: (8,),
             7: (8,),
             8: (10,),
             9: (10,),
             10: (11,),
             11: (12,),
             12: (13,),
             13: (14,),
             14: ()}.
        """
        succ_funcs = [[] for _ in range(DAG_len)]
        for j in range(DAG_len):
            # 获取函数编号
            name_str_list = DAG.loc[j + idx, 'task_name'].strip().split('_')
            name_str_list_len = len(name_str_list)
            func_str_len = len(name_str_list[0])
            func_num = int(name_str_list[0][1:func_str_len])

            if name_str_list_len == 1:  # 如果没有依赖其他函数，则跳过
                pass
            else:  # 如果依赖其他函数，在其他函数中添加后继
                for i in range(name_str_list_len - 1):
                    if name_str_list[i + 1] == '':
                        continue
                    dependent_func_num = int(name_str_list[i + 1])
                    succ_funcs[dependent_func_num - 1].append(func_num)
        succ = dict()
        for i in range(DAG_len):
            succ[i + 1] = tuple(succ_funcs[i])
        del succ_funcs
        return succ

    def get_comp_cost(self, funcs_num, DAG_pp_required):
        """
        获取编号为 funcs_num 在每个 server 上的计算成本
        """
        comp_cost_array = np.zeros((len(funcs_num) + 1, args.n_nodes))
        for i in range(len(funcs_num)):
            func_num = funcs_num[i]
            comp_cost_array[func_num] = DAG_pp_required[func_num - 1] / self.pp
        return comp_cost_array

    @staticmethod
    def comp_cost(job, agent, comp_cost_array):
        """
        计算 job 在 agent 上的计算成本（执行时间）
        """
        a = int(agent)
        return comp_cost_array[job][a]

    @staticmethod
    def w_bar(ni, agents, comp_cost, comp_cost_array):
        """
        计算 task_ni 在所有 agents 中的平均计算成本
        """
        return sum(comp_cost(ni, agent, comp_cost_array) for agent in agents) / len(agents)

    def get_comm_cost(self, succ, DAG_data_stream):
        """
        Get the data transmission cost between any two servers for a given DAG.
        """
        # fix the path chosen between any two node
        fix_path_reciprocals = np.zeros((args.n_nodes, args.n_nodes))
        for n1 in range(args.n_nodes):
            for n2 in range(args.n_nodes):
                if n1 != n2:
                    paths_num = len(self.reciprocal_list[n1][n2])
                    chosen_path = random.randint(0, paths_num - 1)
                    fix_path_reciprocals[n1][n2] = self.reciprocal_list[n1][n2][chosen_path]

        comm_cost_array = []
        for dependent_func_num, funcs_num in succ.items():
            if funcs_num == ():
                pass
            else:
                trans_size = DAG_data_stream[dependent_func_num - 1]
                trans_cost = np.zeros((args.n_nodes, args.n_nodes))  # n * n 的矩阵
                for n1 in range(args.n_nodes):
                    for n2 in range(args.n_nodes):
                        if n1 != n2:
                            trans_cost[n1][n2] = trans_size * fix_path_reciprocals[n1][n2]
                comm_cost_array.append([dependent_func_num, funcs_num, trans_cost])
        del fix_path_reciprocals
        return comm_cost_array

    @staticmethod
    def comm_cost(ni, nj, A, B, comm_cost_array):
        """
        计算 agent_A 上的 job_ni 和 agent_B 上的 job_nj 之间的通信成本。
        """
        a1 = int(A)
        a2 = int(B)
        for d in range(len(comm_cost_array)):
            if ni == comm_cost_array[d][0]:
                funcs_num = comm_cost_array[d][1]
                for f in range(len(funcs_num)):
                    if nj == funcs_num[f]:
                        return comm_cost_array[d][2][a1][a2]
        return 0.

    @staticmethod
    def c_bar(ni, nj, agents, comm_cost, comm_cost_array):
        """
        计算 task_ni 和 task_nj 在所有 agents 之间的平均通信成本
        """
        n = len(agents)
        if n == 1:
            return 0
        n_pairs = args.n_nodes * (args.n_nodes - 1)
        return 1. * sum(
            comm_cost(ni, nj, a1, a2, comm_cost_array) for a1 in agents for a2 in agents if a1 != a2
        ) / n_pairs

    @staticmethod
    def ranku(ni, agents, succ, comp_cost, comm_cost, comp_cost_array, comm_cost_array):
        """
        对 task_ni 计算优先级，公式参考 https://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time
        """
        # 使用偏函数固定某些参数，使调用更方便
        rank = partial(HEFT.ranku, comp_cost=comp_cost, comm_cost=comm_cost,
                       succ=succ, agents=agents, comp_cost_array=comp_cost_array, comm_cost_array=comm_cost_array)
        w = partial(HEFT.w_bar, agents=agents, comp_cost=comp_cost, comp_cost_array=comp_cost_array)
        c = partial(HEFT.c_bar, agents=agents, comm_cost=comm_cost, comm_cost_array=comm_cost_array)

        # 递归调用上面定义的偏函数
        if ni in succ and succ[ni]:  # 若 ni 有后继？这么理解对吗
            return w(ni) + max(c(ni, nj) + rank(nj) for nj in succ[ni])
        else:
            return w(ni)

    @staticmethod
    def end_time(job, events):
        """
        在 events 列表中搜索 job 的结束时间。
        """
        for e in events:
            if e.job == job:
                return e.end

    @staticmethod
    def find_first_gap(agent_orders, desired_start_time, duration):
        """
        将 job 插入到最先匹配的时间间隙中（该间隙的开始时间 <= desired_start_time，且间隙的长度 >= duration）
        返回最早的可插入时间
        """
        # 如果 agent 中没有需要执行的 orders，那么可直接运行 job，无需等待
        if (agent_orders is None) or (len(agent_orders)) == 0:
            return desired_start_time

        # 在每两个相邻的 event 中尝试插入当前任务
        # 在 agent_orders 之前插入一个 dummy event，方便在任何真实 event 开始之前检查间隙
        a = chain([Event(None, None, 0)], agent_orders[:-1])  # chain('ABC', 'DEF') --> A B C D E F
        for e1, e2 in zip(a, agent_orders):  # zip([1, 2], ['sugar', 'spice']) --> (1, 'sugar'), (2, 'spice')
            earliest_start = max(desired_start_time, e1.end)
            if e2.start - earliest_start > duration:
                return earliest_start

        # No gaps found: put it at the end, or whenever the task is ready
        return max(agent_orders[-1].end, desired_start_time)

    @staticmethod
    def start_time(agent, job, orders, jobson, prev, comm_cost, comm_cost_array, comp_cost, comp_cost_array):
        """
        计算 job 能够在 agent 上执行的最早时间
        """
        duration = comp_cost(job, agent, comp_cost_array)  # 获取 job 在 agent 上的计算成本

        # 如果直接前驱中包含 job 说明需要等待前驱执行完毕，才能执行 job
        if job in prev:
            comm_ready = max([HEFT.end_time(p, orders[jobson[p]])  # p 在 jobson[p] 上的完成时间
                              # agent 上的 p 和 jobson[p] 上的 job 的通信成本
                              + comm_cost(p, job, agent, jobson[p], comm_cost_array)
                              for p in prev[job]])
        else:
            comm_ready = 0

        return HEFT.find_first_gap(orders[agent], comm_ready, duration)

    @staticmethod
    def allocate(job, orders, jobson, prev, comm_cost, comm_cost_array, comp_cost, comp_cost_array):
        """
        将 job 分配到能 最早完成 该 job 的 agent 上。
        空间复杂度为 O(1)
        """
        st = partial(HEFT.start_time, job=job, orders=orders, jobson=jobson, prev=prev,
                     comm_cost=comm_cost, comm_cost_array=comm_cost_array,
                     comp_cost=comp_cost, comp_cost_array=comp_cost_array)

        # ft = lambda machine: st(machine) + comp_cost(job, machine, comp_cost_array)
        def ft(machine):
            return st(machine) + comp_cost(job, machine, comp_cost_array)

        agent = min(orders.keys(), key=ft)  # 将 orders.keys() 作为参数传入 ft，选择能最早执行 task 的 agent
        start = st(agent)
        end = ft(agent)

        orders[agent].append(Event(job, start, end))
        orders[agent] = sorted(orders[agent], key=lambda e: e.start)  # 自定义排序规则：按照 e.start 升序
        # Might be better to use a different data structure to keep each
        # agent's orders sorted at a lower cost.

        jobson[job] = agent

    @staticmethod
    def makespan(orders):
        """
        计算最后一个 job 的完成时间。
        """
        return max(v[-1].end for v in orders.values() if v)

    @staticmethod
    def schedule(succ, agents, comp_cost, comp_cost_array, comm_cost, comm_cost_array):
        """
        根据 DAG 将 job 调度到 agents 上。

        :param succ: 若 DAG 中的某个 task = {a: (b, c)} 则 b 和 c 是 a 的直接后继
        :param agents: agents 的集合，可用于执行任务
        :param comp_cost: 这是函数，返回 job 在 agent 上的执行时间
        :param comp_cost_array: 计算成本列表
        :param comm_cost: 这是函数，返回 agent1 上的 job1 和 agent2 上的 job2 之间的通信时间
        :param comm_cost_array: 通信成本列表
        :return: orders: agent 上运行的 job 列表, jobson: 为 job 分配的 agent, makespan: 最后一个任务的完成时间
        """
        rank = partial(HEFT.ranku, agents=agents, succ=succ,
                       comp_cost=comp_cost, comp_cost_array=comp_cost_array,
                       comm_cost=comm_cost, comm_cost_array=comm_cost_array)
        prev = reverse_dict(succ)

        jobs = set(succ.keys()) | set(x for xx in succ.values() for x in xx)
        jobs = sorted(jobs, key=rank)

        orders = {agent: [] for agent in agents}
        print(orders)
        jobson = dict()
        for job in reversed(jobs):
            HEFT.allocate(job, orders, jobson, prev, comm_cost, comm_cost_array, comp_cost, comp_cost_array)

        for n in range(args.n_nodes):
            orders['server ' + str(n + 1)] = orders.pop(str(n))

        return orders, jobson, HEFT.makespan(orders)
