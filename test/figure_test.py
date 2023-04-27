import unittest

import pandas as pd

from models.utils.dag import DAG
from models.utils.dataset import get_one_job
from models.utils.figure import *
from models.utils.udg import UDG


class FigureTest(unittest.TestCase):

    # 图 4-13 问题规模对算法运行时间的影响
    def test_problem_size_figure(self):
        problem_size_figure = ProblemSizeFigure()
        problem_size_figure.visual()

    # 图 4-14 作业完成时间对比
    def test_makespan(self):
        makespan_figure = MakespanFigure()
        makespan_figure.visual(None, None)

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
        idx, selected_dag_id = 0, 0
        selected_dag_name = ['j_11624', 'j_12288', 'j_34819', 'j_54530', 'j_77773', 'j_82634']
        while idx < df.shape[0]:
            job, idx = get_one_job(df, idx)
            G, job_name = DAG.generate_dag_from_alibaba_trace_data(job)
            dag_figure = DAGFigure()
            if selected_dag_id < len(selected_dag_name) and job_name == selected_dag_name[selected_dag_id]:
                letter = chr(ord('a') + selected_dag_id)
                selected_dag_id += 1
                dag_figure.visual(G, '(' + letter + ') ' + job_name)
            else:
                dag_figure.visual(G, job_name)

    # 图 4-1 OpenWhisk对任务的调度方案示例
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

    # 图 3-11 不同函数运行期间使用的资源变化情况
    def test_workload(self):
        df = pd.read_csv(args.selected_container_usage_path)
        workload_figure = WorkloadFigure()
        workload_figure.visual(df, None)

    # 图 3-16 分支预测器的拟合效率
    def test_branch_prediction(self):
        branch_prediction_figure = BranchPredictionFigure()
        branch_prediction_figure.visual(21, None)

    # 图 3-12 DAG重构结果展示
    def test_merge_image1(self):
        image_list = [
            "E:\\Workshop\\real-time-faas\\results\\dags\\(a) j_11624.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\(b) j_12288.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\(c) j_34819.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\(d) j_54530.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\(e) j_77773.png",
            "E:\\Workshop\\real-time-faas\\results\\dags\\(f) j_82634.png"
        ]
        MergeFigure().visual(image_list, None)

    # 图 4-9 随机初始化的节点连通性矩阵
    def test_merge_image2(self):
        image_list = [
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(a) 10 nodes 5 connections.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(b) 10 nodes 2 connections.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(c) 10 nodes 3 connections.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(d) 10 nodes 5 connections.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(e) 10 nodes 3 connections.png",
            "E:\\Workshop\\real-time-faas\\results\\udgs\\(f) 10 nodes 5 connections.png",
        ]
        MergeFigure().visual(image_list, None)

    # 图 4-9 随机初始化的节点连通性矩阵
    def test_udg_generation(self):
        udg_figure = UDGFigure()
        udg = UDG()

        G, udg_name = udg.generate_udg_from_random(10, 5, 20, 60)
        udg_figure.visual(G, '(a) ' + udg_name)
        G, udg_name = udg.generate_udg_from_random(10, 2, 20, 60)
        udg_figure.visual(G, '(b) ' + udg_name)
        G, udg_name = udg.generate_udg_from_random(10, 3, 20, 60)
        udg_figure.visual(G, '(c) ' + udg_name)
        G, udg_name = udg.generate_udg_from_random(10, 5, 20, 60)
        udg_figure.visual(G, '(d) ' + udg_name)
        G, udg_name = udg.generate_udg_from_random(10, 3, 20, 60)
        udg_figure.visual(G, '(e) ' + udg_name)
        G, udg_name = udg.generate_udg_from_random(10, 5, 20, 60)
        udg_figure.visual(G, '(f) ' + udg_name)

    def test_chinese_display(self):
        chinese_text_figure = ChineseTextFigure()
        chinese_text_figure.visual()


if __name__ == '__main__':
    unittest.main()
