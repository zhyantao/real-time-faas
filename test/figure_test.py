import unittest

import pandas as pd

from models.utils.dag import DAG
from models.utils.dataset import get_one_job
from models.utils.figure import *
from models.utils.udg import UDG


class FigureTest(unittest.TestCase):

    def test_problem_size_figure(self):
        problem_size_figure = ProblemSizeFigure()
        problem_size_figure.visual()

    def test_runtime_figure(self):
        runtime_figure = RuntimeFigure()
        runtime_figure.visual()

    def test_gcn_params_figure(self):
        gcn_layer_figure = GCNParamsFigure()
        gcn_layer_figure.visual()

    def test_dqn_params_figure(self):
        dqn_params_figure = DQNParamsFigure()
        dqn_params_figure.visual()

    def test_workload_analysis_figure(self):
        df = pd.read_csv(args.selected_container_usage_path)
        workload_analysis_figure = WorkloadAnalysisFigure()
        workload_analysis_figure.visual(df, None)

    def test_rebuild_DAG(self):
        # 重建 DAG
        df = pd.read_csv(args.selected_batch_task_path)
        idx = 0
        while idx < df.shape[0]:
            job, idx = get_one_job(df, idx)
            G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
            dag_figure = DAGFigure()
            dag_figure.visual(G, job_name)

    def test_gantt(self):
        mappings = {
            0: [ScheduleEvent(task_id=10, start=0, end=14.0, cpu_id=0),
                ScheduleEvent(task_id=13, start=14.0, end=27.0, cpu_id=0),
                ScheduleEvent(task_id=1, start=27.0, end=40.0, cpu_id=0),
                ScheduleEvent(task_id=12, start=40.0, end=51.0, cpu_id=0),
                ScheduleEvent(task_id=7, start=57.0, end=62.0, cpu_id=0),
                ScheduleEvent(task_id=15, start=62.0, end=75.0, cpu_id=0),
                ScheduleEvent(task_id=16, start=75.0, end=82.0, cpu_id=0),
                ScheduleEvent(task_id=17, start=86.0, end=91.0, cpu_id=0)],
            1: [ScheduleEvent(task_id=3, start=18.0, end=26.0, cpu_id=1),
                ScheduleEvent(task_id=5, start=26.0, end=42.0, cpu_id=1),
                ScheduleEvent(task_id=14, start=42.0, end=55.0, cpu_id=1),
                ScheduleEvent(task_id=8, start=56.0, end=68.0, cpu_id=1),
                ScheduleEvent(task_id=9, start=73.0, end=80.0, cpu_id=1),
                ScheduleEvent(task_id=19, start=102.0, end=109.0, cpu_id=1)],
            2: [ScheduleEvent(task_id=0, start=0, end=9.0, cpu_id=2),
                ScheduleEvent(task_id=2, start=9.0, end=28.0, cpu_id=2),
                ScheduleEvent(task_id=4, start=28.0, end=38.0, cpu_id=2),
                ScheduleEvent(task_id=6, start=38.0, end=49.0, cpu_id=2),
                ScheduleEvent(task_id=11, start=49.0, end=67.0, cpu_id=2),
                ScheduleEvent(task_id=18, start=68.0, end=88.0, cpu_id=2)]
        }

        gantt_figure = GanttFigure()
        gantt_figure.visual(mappings, None)

    def test_workload(self):
        df = pd.read_csv(args.selected_container_usage_path)
        workload_figure = WorkloadFigure()
        workload_figure.visual(df, None)

    def test_branch_prediction(self):
        branch_prediction_figure = BranchPredictionFigure()
        branch_prediction_figure.visual(21, None)

    def test_makespan(self):
        makespan_figure = MakespanFigure()
        makespan_figure.visual(None, None)

    def test_merge_images(self):
        image_list1 = [
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_11624.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_12288.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_34819.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_54530.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_77773.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\j_82634.png"
        ]
        MergeFigure().visual(image_list1, None)

        image_list2 = [
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_3_1680084850.4353304.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_2_1680084824.861884.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_3_1680084851.3031876.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_3_1680084852.2823596.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_5_1680084863.1467211.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\max_connection_5_1680084863.8470132.png",
        ]
        MergeFigure().visual(image_list2, None)

    def test_udg_generation(self):
        udg_figure = UDGFigure()
        udg = UDG()

        G, udg_name = udg.generate_udg_from_random(10, 5, 20, 60)
        udg_figure.visual(G, udg_name + '_' + str(time.time()))


if __name__ == '__main__':
    unittest.main()
