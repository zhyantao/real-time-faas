"""
Step 1: Get the topological order of DAGs.
Step 2: Generate the scenario.
Step 3: Run the three algorithms and compare the results.
"""
from models.utils.dataset_processing import sample_DAG, get_topological_order
from models.utils.scenario import *
from models.scheduler.dpe import DPE
from models.scheduler.fixdoc import FixDoc
from models.scheduler.heft2 import HEFT
from models.utils.interpretate_result import *
from models.utils.parameters import *
import datetime


if __name__ == '__main__':
    print('------------------------ Step 1 ------------------------')
    sample_DAG()
    get_topological_order()

    print('\n\n------------------------ Step 2 ------------------------')
    G, bw, pp = generate_scenario()
    print_scenario(G, bw, pp)
    simple_paths = get_simple_paths(G)
    print_simple_paths(simple_paths)
    reciprocals_list, proportions_list = get_ratio(simple_paths, bw)
    pp_required, data_stream = set_funcs()

    print('\n\n------------------------ Step 3 ------------------------')
    dpe = DPE(G, bw, pp, simple_paths, reciprocals_list, proportions_list, pp_required, data_stream)
    start = datetime.datetime.now()
    T_optimal_all_dpe, DAGs_deploy_dpe, process_sequence_all_dpe, start_time_all_dpe = dpe.get_response_time(sorted_DAG_path=SORTED_DAG_PATH)
    end = datetime.datetime.now()
    print('Computer\'s running time:', (end - start).seconds, 'seconds')
    DAG_num_chosen = 2010    # 随机选择采样出来的 DAG 中的某一个
    # show_DAG()  # TODO
    print_scheduling_results(T_optimal_all_dpe, DAGs_deploy_dpe, process_sequence_all_dpe, start_time_all_dpe, DAG_num_chosen)

    fixdoc = FixDoc(G, bw, pp, simple_paths, reciprocals_list, proportions_list, pp_required, data_stream)
    start = datetime.datetime.now()
    T_optimal_all_fixdoc, DAGs_deploy_fixdoc, process_sequence_all_fixdoc, start_time_all_fixdoc = fixdoc.get_response_time(sorted_DAG_path=SORTED_DAG_PATH)
    end = datetime.datetime.now()
    print('Computer\'s running time:', (end - start).seconds, 'seconds')
    print_scheduling_results(T_optimal_all_fixdoc, DAGs_deploy_fixdoc, process_sequence_all_fixdoc, start_time_all_fixdoc, DAG_num_chosen)

    heft = HEFT(G, bw, pp, simple_paths, reciprocals_list, proportions_list, pp_required, data_stream)
    start = datetime.datetime.now()
    DAGs_orders, DAGs_deploy = heft.get_response_time(sorted_DAG_path=SORTED_DAG_PATH)
    end = datetime.datetime.now()
    print('Computer\'s running time:', (end - start).seconds, 'seconds')
    print('\nThe finish time of each function on the chosen server for DAG #%d:' % DAG_num_chosen)
    pprint.pprint(DAGs_orders[DAG_num_chosen])
