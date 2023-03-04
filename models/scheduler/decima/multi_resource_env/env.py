from multi_resource_env.executor import MultiResExecutor as Executor
from multi_resource_env.executor_commit import MultiResExecutorCommit as ExecutorCommit
from multi_resource_env.group_executors import group_executors
from multi_resource_env.job_generator import generate_jobs
from multi_resource_env.node_selected import NodeSelected
from multi_resource_env.task import MultiResTask as Task
from param import *
from spark_env.action_map import compute_act_map
from spark_env.env import Environment as SparkEnvironment
from spark_env.free_executors import FreeExecutors
from spark_env.job_dag import JobDAG
from utils import *


class MultiResEnvironment(SparkEnvironment):
    def __init__(self):
        SparkEnvironment.__init__(self)

        # overwrite executors
        assert len(args.exec_group_num) == len(args.exec_cpus)
        assert len(args.exec_group_num) == len(args.exec_mems)
        self.executors = OrderedSet()
        exec_id = 0
        for i in range(len(args.exec_group_num)):
            exec_num = args.exec_group_num[i]
            exec_cpu = args.exec_cpus[i]
            exec_mem = args.exec_mems[i]
            for _ in range(exec_num):
                self.executors.add(Executor(exec_id, i, exec_cpu, exec_mem))
                exec_id += 1

        # overwrite free executors
        self.free_executors = FreeExecutors(self.executors)

        # overwrite executor commit
        self.exec_commit = ExecutorCommit()

        # overwrite node selected
        self.node_selected = NodeSelected(len(args.exec_group_num))

    def assign_executor(self, executor, frontier_changed):
        # overwrite how we represent available executors
        if executor.node is not None and not executor.node.no_more_tasks:
            # keep working on the previous node
            task = executor.node.schedule(executor)
            self.timeline.push(task.finish_time, task)
        else:
            # need to move on to other nodes
            if frontier_changed:
                # frontier changed, need to consult all free executors
                # note: executor.job_dag might change after self.schedule()
                source_job = executor.job_dag
                if self.exec_commit.get_len(
                        executor.type, executor.node) > 0:
                    # directly fulfill the commitment
                    self.exec_to_schedule = {executor}
                    self.schedule()
                else:
                    # free up the executor
                    self.free_executors.add(source_job, executor)
                # then consult all free executors
                self.exec_to_schedule = OrderedSet(
                    self.free_executors[source_job])
                self.source_job = source_job
                self.num_source_exec = group_executors(
                    self.free_executors[source_job],
                    len(args.exec_group_num))
            else:
                # just need to schedule one current executor
                self.exec_to_schedule = {executor}
                # only care about executors on the node
                if self.exec_commit.get_len(
                        executor.type, executor.node) > 0:
                    # directly fulfill the commitment
                    self.schedule()
                else:
                    # need to consult for ALL executors on the node
                    # Note: self.exec_to_schedule is immediate
                    #       self.num_source_exec is for commit
                    #       so len(self.exec_to_schedule) !=
                    #       self.num_source_exec can happen
                    self.source_job = executor.job_dag
                    self.num_source_exec = \
                        group_executors(executor.node.executors,
                                        len(args.exec_group_num))

    def backup_schedule(self, executor):
        backup_scheduled = False
        if executor.job_dag is not None:
            # first try to schedule on current job
            for node in executor.job_dag.frontier_nodes:
                if not self.saturated(node) and \
                        executor.cpu >= node.cpu and \
                        executor.mem >= node.mem:
                    # greedily schedule a frontier node
                    task = node.schedule(executor)
                    self.timeline.push(task.finish_time, task)
                    backup_scheduled = True
                    break
        # then try to schedule on any available node
        if not backup_scheduled:
            num_source_exec = [0 for _ in range(len(args.exec_group_num))]
            num_source_exec[executor.type] = 1
            # only care about the specific executor type
            schedulable_nodes = \
                self.get_frontier_nodes(num_source_exec)[executor.type]
            if len(schedulable_nodes) > 0:
                node = next(iter(schedulable_nodes))
                self.timeline.push(
                    self.wall_time.curr_time + args.moving_delay, executor)
                # keep track of moving executors
                self.moving_executors.add(executor, node)
                backup_scheduled = True
        # at this point if nothing available, leave executor idle
        if not backup_scheduled:
            self.free_executors.add(executor.job_dag, executor)

    def get_frontier_nodes(self, num_source_execs):
        # overwrite how frontier nodes are computed
        # with executor types
        # Note: the frontier node is with respect to 
        # different kinds of executors
        assert len(num_source_execs) == len(args.exec_group_num)
        frontier_nodes = {i: OrderedSet() \
                          for i in range(len(num_source_execs))}

        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                # check different executor types
                for i in range(len(num_source_execs)):
                    if not node in self.node_selected[i] \
                            and not self.saturated(node):
                        parents_saturated = True
                        for parent_node in node.parent_nodes:
                            if not self.saturated(parent_node):
                                parents_saturated = False
                                break
                        if parents_saturated:
                            # check if node fits this executors type
                            if num_source_execs[i] > 0 and \
                                    node.cpu <= args.exec_cpus[i] and \
                                    node.mem <= args.exec_mems[i]:
                                frontier_nodes[i].add(node)

        return frontier_nodes

    def observe(self):
        return self.job_dags, self.source_job, self.num_source_exec, \
            self.get_frontier_nodes(self.num_source_exec), \
            self.exec_commit, self.moving_executors, self.action_map

    def schedule(self):
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # schedule executors from the source until the commitment is fulfilled
        while len(self.exec_to_schedule) > 0:

            # keep fulfilling the commitment using free executors
            executor = self.exec_to_schedule.pop()
            node = self.exec_commit.pop(executor.type, source)

            # mark executor as in use if it was free executor previously
            if self.free_executors.contain_executor(executor.job_dag, executor):
                self.free_executors.remove(executor)

            if node is None:
                # the next node is explicitly silent, make executor ilde
                if executor.job_dag is not None and \
                        any([not n.no_more_tasks for n in \
                             executor.job_dag.nodes]):
                    # mark executor as idle in its original job
                    self.free_executors.add(executor.job_dag, executor)
                else:
                    # no where to assign, put executor in null pool
                    self.free_executors.add(None, executor)


            elif not node.no_more_tasks:
                # node is not currently saturated
                if executor.job_dag == node.job_dag:
                    # executor local to the job
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # put executor back in the free pool
                        self.free_executors.add(executor.job_dag, executor)

                else:
                    # need to move executor
                    self.timeline.push(
                        self.wall_time.curr_time + args.moving_delay, executor)
                    # keep track of moving executors
                    self.moving_executors.add(executor, node)

            else:
                # node is already saturated, use backup logic
                self.backup_schedule(executor)

    def step(self, next_node, exec_type, limit):

        # mark the node as selected
        assert next_node not in self.node_selected[exec_type]
        self.node_selected[exec_type].add(next_node)
        # commit the source executor
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # compute number of valid executors to assign
        if next_node is not None:
            use_exec = min(next_node.num_tasks - next_node.next_task_idx - \
                           self.exec_commit.node_commit[next_node] - \
                           self.moving_executors.count(next_node), limit)
        else:
            use_exec = limit
        assert use_exec > 0

        self.exec_commit.add(source, next_node, exec_type, use_exec)
        # deduct the executors that know the destination
        self.num_source_exec[exec_type] -= use_exec
        assert self.num_source_exec[exec_type] >= 0

        if sum(self.num_source_exec) == 0:
            # now a new scheduling round, clean up node selection
            self.node_selected.clear()
            # all commitments are made, now schedule free executors
            self.schedule()

        # Now run to the next event in the virtual timeline
        while len(self.timeline) > 0 and sum(self.num_source_exec) == 0:
            # consult agent by putting executors in source_exec

            new_time, obj = self.timeline.pop()
            self.wall_time.update_time(new_time)

            # case task: a task completion event, and frees up an executor.
            # case query: a new job arrives
            # case executor: an executor arrives at certain job

            if isinstance(obj, Task):  # task completion event
                finished_task = obj
                node = finished_task.node
                node.num_finished_tasks += 1

                # bookkeepings for node completion
                frontier_changed = False
                if node.num_finished_tasks == node.num_tasks:
                    assert not node.tasks_all_done  # only complete once
                    node.tasks_all_done = True
                    node.job_dag.num_nodes_done += 1
                    node.node_finish_time = self.wall_time.curr_time

                    frontier_changed = node.job_dag.update_frontier_nodes(node)

                # assign new destination for the job
                self.assign_executor(finished_task.executor, frontier_changed)

                # bookkeepings for job completion
                if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
                    assert not node.job_dag.completed  # only complete once
                    node.job_dag.completed = True
                    node.job_dag.completion_time = self.wall_time.curr_time
                    self.remove_job(node.job_dag)

            elif isinstance(obj, JobDAG):  # new job arrival event
                job_dag = obj
                # job should be arrived at the first time
                assert not job_dag.arrived
                job_dag.arrived = True
                # inform agent about job arrival when stream is enabled
                self.job_dags.add(job_dag)
                self.add_job(job_dag)
                self.action_map = compute_act_map(self.job_dags)
                # assign free executors (if any) to the new job
                if len(self.free_executors[None]) > 0:
                    self.exec_to_schedule = \
                        OrderedSet(self.free_executors[None])
                    self.source_job = None
                    self.num_source_exec = \
                        group_executors(
                            self.free_executors[None],
                            len(args.exec_group_num))

            elif isinstance(obj, Executor):  # executor arrival event
                executor = obj
                # pop destination from the tracking record
                node = self.moving_executors.pop(executor)

                if node is not None:
                    # the job is not yet done when executor arrives
                    executor.job_dag = node.job_dag
                    node.job_dag.executors.add(executor)

                if node is not None and not node.no_more_tasks:
                    # the node is still schedulable
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # free up the executor in this job
                        self.free_executors.add(executor.job_dag, executor)
                else:
                    # the node is saturated or the job is done
                    # by the time the executor arrives, use
                    # backup logic
                    self.backup_schedule(executor)

            else:
                print("illegal event type")
                exit(1)

        # compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)

        # no more decision to make, jobs all done or time is up
        done = (sum(self.num_source_exec) == 0) and \
               ((len(self.timeline) == 0) or \
                (self.wall_time.curr_time >= self.max_time))

        if done:
            assert self.wall_time.curr_time >= self.max_time or \
                   len(self.job_dags) == 0

        return self.observe(), reward, done

    def reset(self, max_time=np.inf):
        self.max_time = max_time
        self.wall_time.reset()
        self.timeline.reset()
        self.exec_commit.reset()
        self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.node_selected.clear()
        for executor in self.executors:
            executor.reset()
        self.free_executors.reset(self.executors)
        # overwrite the generation a set of new jobs
        self.job_dags = generate_jobs(
            self.np_random, self.timeline, self.wall_time)
        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        # add initial set of jobs in the system
        for job_dag in self.job_dags:
            self.add_job(job_dag)
        # put all executors as source executors initially
        self.source_job = None
        self.num_source_exec = group_executors(
            self.executors, len(args.exec_group_num))
        self.exec_to_schedule = OrderedSet(self.executors)
