from models.scheduler.dpe import DPE
from models.scheduler.heft import HEFT
from models.utils.dataset import sample_jobs, get_topological_order
from models.utils.params import args
from models.scheduler.scenario import generate_scenario, print_scenario, get_simple_paths, \
    print_simple_paths, get_ratio, set_funcs

if __name__ == '__main__':
    # 采集数据，并对 task 进行拓扑排序
    sample_jobs()
    get_topological_order()

    dpe_makespan = []
    heft_makespan = []

    # 生成边缘计算场景
    G, bw, pp = generate_scenario()
    print_scenario(G, bw, pp)
    simple_paths = get_simple_paths(G)
    print_simple_paths(simple_paths)
    reciprocals_list, proportions_list = get_ratio(simple_paths, bw)
    pp_required, data_stream = set_funcs()

    # 进行算法对比 DPE
    dpe = DPE(G, bw, pp, simple_paths, reciprocals_list, proportions_list, pp_required, data_stream)
    # start = datetime.datetime.now()
    cpu_earliest_finish_time_all_dpe, task_deployment_all_dpe, \
        cpu_task_mapping_list_all_dpe, task_start_time_all_dpe, makespan_avg_dpe \
        = dpe.get_response_time(sorted_job_path=args.batch_task_topological_order_path)
    # print(task_deployment_all_dpe)
    # end = datetime.datetime.now()
    # print('Computer\'s running time:', (end - start).microseconds, 'microseconds')
    # show_DAG()  # TODO
    # scheduling_result = SchedulingResult(cpu_earliest_finish_time_all_dpe,
    #                                      task_deployment_all_dpe,
    #                                      cpu_task_mapping_list_all_dpe,
    #                                      task_start_time_all_dpe,
    #                                      job_num_chosen)
    # scheduling_result.print()
    dpe_makespan.append(makespan_avg_dpe)

    # 进行算法对比 HEFT
    heft = HEFT(G, bw, pp, simple_paths, reciprocals_list, proportions_list, pp_required, data_stream)
    # start = datetime.datetime.now()
    cpu_task_mapping_list_all, task_deployment_all, makespan_avg_heft \
        = heft.get_response_time(sorted_job_path=args.batch_task_topological_order_path)
    # end = datetime.datetime.now()
    # print('Computer\'s running time:', (end - start).microseconds, 'microseconds')
    # print('\nThe finish time of each task on the chosen cpu for job #%d:' % job_num_chosen)
    # pprint.pprint(cpu_task_mapping_list_all[job_num_chosen])
    heft_makespan.append(makespan_avg_heft)

    # plt.plot(range(50), dpe_makespan, label='DPE')
    # plt.plot(range(50), heft_makespan, label='HEFT')
    # plt.suptitle('DPE vs. HEFT')
    # plt.ylabel("Makespan of Job (s)")
    # plt.xlabel("Job number")
    # plt.legend()
    # plt.show()
