def group_executors(executors, num_types):
    num_exec = [0 for _ in range(num_types)]
    for executor in executors:
        num_exec[executor.type] += 1
    return num_exec
