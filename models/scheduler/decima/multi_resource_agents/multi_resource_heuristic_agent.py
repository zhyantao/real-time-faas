import numpy as np

from agent import Agent
from multi_resource_env.node import MultiResNode as Node
from param import *
from spark_env.job_dag import JobDAG


class MultiResDynamicPartitionAgent(Agent):
    # dynamically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self):
        Agent.__init__(self)

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
            frontier_nodes, exec_commit, \
            moving_executors, action_map = obs

        # there should be some executors to assign
        assert sum(num_source_exec) > 0

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([any(n.next_task_idx + \
                                       exec_commit.node_commit[n] + moving_executors.count(n) \
                                       < n.num_tasks for n in job_dag.nodes) \
                                   for job_dag in job_dags])

        # compute the executor cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for exec_commit_of_kind in exec_commit.commit:
            for s in exec_commit_of_kind:
                if isinstance(s, JobDAG):
                    j = s
                elif isinstance(s, Node):
                    j = s.job_dag
                elif s is None:
                    j = None
                else:
                    print('source', s, 'unknown')
                    exit(1)
                for n in exec_commit_of_kind[s]:
                    if n is not None and n.job_dag != j:
                        exec_map[n.job_dag] += exec_commit_of_kind[s][n]

        assert len(frontier_nodes) == len(num_source_exec)

        for i in range(len(num_source_exec)):
            # skip if number of source executor of this kind is 0
            if num_source_exec[i] == 0:
                continue

            scheduled = False
            # first assign executor to the same job
            if source_job is not None:
                # immediately scheduable nodes
                for node in source_job.frontier_nodes:
                    if node in frontier_nodes[i]:
                        return node, i, num_source_exec[i]
                # schedulable node in the job
                for node in frontier_nodes[i]:
                    if node.job_dag == source_job:
                        return node, i, num_source_exec[i]

            # the source job is finished or does not exist
            for job_dag in job_dags:
                if exec_map[job_dag] < exec_cap:
                    next_node = None
                    # immediately scheduable node first
                    for node in job_dag.frontier_nodes:
                        if node in frontier_nodes[i]:
                            next_node = node
                            break
                    # then schedulable node in the job
                    if next_node is None:
                        for node in frontier_nodes:
                            if node in job_dag.nodes:
                                next_node = node
                                break
                    # node is selected, compute limit
                    if next_node is not None:
                        use_exec = min(
                            next_node.num_tasks - next_node.next_task_idx - \
                            exec_commit.node_commit[next_node] - \
                            moving_executors.count(next_node),
                            exec_cap - exec_map[job_dag],
                            num_source_exec[i])
                        return next_node, i, use_exec

        # there is more executors than tasks in the system
        for i in range(len(num_source_exec)):
            if num_source_exec[i] > 0:
                return None, i, num_source_exec[i]


class MultiResPackingAgent(Agent):
    # greedy packing resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self):
        Agent.__init__(self)

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
            frontier_nodes, exec_commit, \
            moving_executors, action_map = obs

        # there should be some executors to assign
        assert sum(num_source_exec) > 0

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([any(n.next_task_idx + \
                                       exec_commit.node_commit[n] + moving_executors.count(n) \
                                       < n.num_tasks for n in job_dag.nodes) \
                                   for job_dag in job_dags])

        # compute the executor cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for exec_commit_of_kind in exec_commit.commit:
            for s in exec_commit_of_kind:
                if isinstance(s, JobDAG):
                    j = s
                elif isinstance(s, Node):
                    j = s.job_dag
                elif s is None:
                    j = None
                else:
                    print('source', s, 'unknown')
                    exit(1)
                for n in exec_commit_of_kind[s]:
                    if n is not None and n.job_dag != j:
                        exec_map[n.job_dag] += exec_commit_of_kind[s][n]

        assert len(frontier_nodes) == len(num_source_exec)

        for i in range(len(num_source_exec)):
            # skip if number of source executor of this kind is 0
            if num_source_exec[i] == 0:
                continue

            # first assign executor to the same job
            if source_job is not None:
                # immediately scheduable nodes
                for node in source_job.frontier_nodes:
                    if node in frontier_nodes[i]:
                        return node, i, num_source_exec[i]
                # schedulable node in the job
                return_node = None
                max_pack_score = -np.inf
                for node in frontier_nodes[i]:
                    if node.job_dag == source_job:
                        pack_score = args.exec_cpus[i] * node.cpu + \
                                     args.exec_mems[i] * node.mem
                        if pack_score > max_pack_score:
                            max_pack_score = pack_score
                            return_node = node
                if return_node is not None:
                    return return_node, i, num_source_exec[i]

        # the source job is finished or does not exist
        return_node = None
        return_type = None
        return_use_exec = None
        max_pack_score = -np.inf
        for i in range(len(num_source_exec)):
            for job_dag in job_dags:
                if exec_map[job_dag] < exec_cap:
                    next_node = None
                    # immediately scheduable node first
                    for node in job_dag.frontier_nodes:
                        if node in frontier_nodes[i]:
                            next_node = node
                            break
                    # then schedulable node in the job
                    if next_node is None:
                        for node in frontier_nodes:
                            if node in job_dag.nodes:
                                next_node = node
                                break
                    # node is selected, compute limit
                    if next_node is not None:
                        use_exec = min(
                            next_node.num_tasks - next_node.next_task_idx - \
                            exec_commit.node_commit[next_node] - \
                            moving_executors.count(next_node),
                            exec_cap - exec_map[job_dag],
                            num_source_exec[i])

                        pack_score = args.exec_cpus[i] * next_node.cpu + \
                                     args.exec_mems[i] * next_node.mem

                        if pack_score > max_pack_score:
                            max_pack_score = pack_score
                            return_node = next_node
                            return_type = i
                            return_use_exec = use_exec

        if return_node is not None:
            return return_node, return_type, return_use_exec

        # there is more executors than tasks in the system
        for i in range(len(num_source_exec)):
            if num_source_exec[i] > 0:
                return None, i, num_source_exec[i]


class MultiResGrapheneAgent(Agent):
    # mimicking graphene scheduling algorithm
    # https://www.usenix.org/conference/osdi16/technical-sessions/presentation/grandl_graphene
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self):
        Agent.__init__(self)

    def get_action(self, obs):

        long_ratio = 1  # long score for difficult tasks
        frag_ratio = 1  # fragmentation score for difficult tasks
        pack_ratio = 1  # pack score for the next task
        dura_ratio = 1  # duration score for favoring small jobs

        # parse observation
        job_dags, source_job, num_source_exec, \
            frontier_nodes, exec_commit, \
            moving_executors, action_map = obs

        # there should be some executors to assign
        assert sum(num_source_exec) > 0

        # explicitly compute unfinished jobs
        num_unfinished_jobs = sum([any(n.next_task_idx + \
                                       exec_commit.node_commit[n] + moving_executors.count(n) \
                                       < n.num_tasks for n in job_dag.nodes) \
                                   for job_dag in job_dags])

        # compute the executor cap
        exec_cap = int(np.ceil(args.exec_cap / max(1, num_unfinished_jobs)))

        # sort out the exec_map
        exec_map = {}
        for job_dag in job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for exec_commit_of_kind in exec_commit.commit:
            for s in exec_commit_of_kind:
                if isinstance(s, JobDAG):
                    j = s
                elif isinstance(s, Node):
                    j = s.job_dag
                elif s is None:
                    j = None
                else:
                    print('source', s, 'unknown')
                    exit(1)
                for n in exec_commit_of_kind[s]:
                    if n is not None and n.job_dag != j:
                        exec_map[n.job_dag] += exec_commit_of_kind[s][n]

        assert len(frontier_nodes) == len(num_source_exec)

        for i in range(len(num_source_exec)):
            # skip if number of source executor of this kind is 0
            if num_source_exec[i] == 0:
                continue

            # first assign executor to the same job
            if source_job is not None:
                # immediately scheduable nodes
                for node in source_job.frontier_nodes:
                    if node in frontier_nodes[i]:
                        return node, i, num_source_exec[i]
                # schedulable node in the job
                return_node = None
                max_pack_score = -np.inf
                for node in frontier_nodes[i]:
                    if node.job_dag == source_job:
                        pack_score = args.exec_cpus[i] * node.cpu + \
                                     args.exec_mems[i] * node.mem
                        if pack_score > max_pack_score:
                            max_pack_score = pack_score
                            return_node = node
                if return_node is not None:
                    return return_node, i, num_source_exec[i]

        # get all work left on all the job
        work_left = {}
        for job_dag in job_dags:
            work_left[job_dag] = 1e-6
            for node in job_dag.nodes:
                if not node.no_more_tasks:
                    work_left[job_dag] += \
                        node.tasks[-1].duration * \
                        (node.num_tasks - node.next_task_idx)

        # the source job is finished or does not exist
        return_node = None
        return_type = None
        return_use_exec = None
        best_score = -np.inf
        for i in range(len(num_source_exec)):
            for job_dag in job_dags:
                if exec_map[job_dag] < exec_cap:
                    next_node = None
                    # immediately scheduable node first
                    for node in job_dag.frontier_nodes:
                        if node in frontier_nodes[i]:
                            next_node = node
                            break
                    # then schedulable node in the job
                    if next_node is None:
                        for node in frontier_nodes:
                            if node in job_dag.nodes:
                                next_node = node
                                break
                    # node is selected, compute limit
                    if next_node is not None:
                        use_exec = min(
                            next_node.num_tasks - next_node.next_task_idx - \
                            exec_commit.node_commit[next_node] - \
                            moving_executors.count(next_node),
                            exec_cap - exec_map[job_dag],
                            num_source_exec[i])

                        pack_score = args.exec_cpus[i] * next_node.cpu + \
                                     args.exec_mems[i] * next_node.mem

                        intrinsic_waves = sum(np.array(args.exec_group_num) * \
                                              (np.array(args.exec_cpus) >= node.cpu) * \
                                              (np.array(args.exec_mems) >= node.mem))

                        schedule_score = \
                            (long_ratio * (next_node.tasks[-1].duration) + \
                             frag_ratio * (next_node.tasks[-1].duration / intrinsic_waves)) / \
                            work_left[job_dag] * dura_ratio + \
                            pack_ratio * pack_score

                        if schedule_score > best_score:
                            best_score = schedule_score
                            return_node = next_node
                            return_type = i
                            return_use_exec = use_exec

        if return_node is not None:
            return return_node, return_type, return_use_exec

        # there is more executors than tasks in the system
        for i in range(len(num_source_exec)):
            if num_source_exec[i] > 0:
                return None, i, num_source_exec[i]
