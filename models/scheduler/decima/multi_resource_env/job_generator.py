from multi_resource_env.node import MultiResNode as Node
from multi_resource_env.task import MultiResTask as Task
from spark_env.job_dag import *
from spark_env.job_generator import pre_process_task_duration, recursive_find_descendant


# overwrite spark_env job_generator's load_job
def load_job(file_path, query_size, query_idx, wall_time, np_random):
    query_path = file_path + query_size + '/'

    adj_mat = np.load(
        query_path + 'adj_mat_' + str(query_idx) + '.npy')
    task_durations = np.load(
        query_path + 'task_duration_' + str(query_idx) + '.npy', allow_pickle=True).item()

    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert adj_mat.shape[0] == len(task_durations)

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l])

        # generate random memory requirement
        task_cpu = 1.0
        task_mem = np_random.uniform(0, 1.0)

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, task_cpu, task_mem, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, task_cpu, task_mem, tasks,
                    task_duration, wall_time, np_random)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, adj_mat,
                     args.query_type + '-' + query_size + '-' + str(query_idx))

    return job_dag


def alibaba_load_job(query_name, query_idx, wall_time, np_random):
    try:
        adj_mat = np.load('./multi_resource_env/alibaba/dags/adj_mat_' + str(query_idx) + '.npy')
        task_durations = np.load('./multi_resource_env/alibaba/dags/task_duration_'
                                 + str(query_idx) + '.npy', encoding='latin1').item()
    except IOError:
        print("Logs of the query " + query + " not exist")
        exit(1)

    assert adj_mat.shape[0] == adj_mat.shape[1]

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):

        task_duration = {}
        if 'first_wave' in task_durations:
            task_duration['first_wave'] = task_durations['first_wave'][n + 1]
        else:
            task_duration['first_wave'] = {}

        for key in task_duration['first_wave']:
            task_duration['first_wave'][key] = list(map(float, task_duration['first_wave'][key]))

        task_duration['rest_wave'] = task_duration['first_wave']
        task_duration['fresh_durations'] = task_duration['first_wave']

        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e])
        rough_duration = np.mean(task_duration['first_wave'][e])

        # memory requirement
        task_cpu = 1.0
        try:
            task_mem = float(task_durations['memory'][n + 1]) / 1000.0
        except:
            task_mem = 0.0

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, task_cpu, task_mem, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, task_cpu, task_mem, tasks,
                    task_duration, wall_time, np_random)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, adj_mat, query_name)

    return job_dag


def generate_alibaba_jobs(np_random, timeline, wall_time):
    job_dags = OrderedSet()
    t = 0

    with open('./multi_resource_env/alibaba/alibaba_valid_file_ids.txt', 'r') as f:
        valid_jobs = [line.rstrip('\n') for line in f]

    job_name_to_id = {}
    for job in valid_jobs:
        job_name = job.split(' ')[0]
        job_id = int(job.split(' ')[1])
        job_name_to_id[job_name] = job_id

    with open('./multi_resource_env/alibaba/alibaba_job_arrival_trace.txt', 'r') as f:
        arrivals = [line.rstrip('\n') for line in f]

    t_init = int(arrivals[0].split(' ')[0])

    for i in range(args.num_stream_dags):
        line = arrivals[i]
        arrival_time = int(line.split(' ')[0]) - t_init
        t += arrival_time

        query_name = line.split(' ')[1]
        query_idx = job_name_to_id[query_name]

        job_dag = alibaba_load_job(query_name, query_idx, wall_time, np_random)

        job_dag.start_time = t
        if t == 0:
            job_dag.arrived = True
            job_dags.add(job_dag)
        else:
            timeline.push(t, job_dag)

    return job_dags


def generate_tpch_jobs(np_random, timeline, wall_time):
    job_dags = OrderedSet()
    t = 0

    for _ in range(args.num_init_dags):
        # generate query
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    for _ in range(args.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(args.stream_interval))
        # uniform distribution
        query_size = args.tpch_size[np_random.randint(len(args.tpch_size))]
        query_idx = str(np_random.randint(args.tpch_num) + 1)
        # generate job
        job_dag = load_job(
            args.job_folder, query_size, query_idx, wall_time, np_random)
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags


def generate_jobs(np_random, timeline, wall_time):
    if args.query_type == 'tpch':
        job_dags = generate_tpch_jobs(np_random, timeline, wall_time)

    elif args.query_type == 'alibaba':
        job_dags = generate_alibaba_jobs(np_random, timeline, wall_time)

    else:
        print('Invalid query type ' + args.query_type)
        exit(1)

    return job_dags
