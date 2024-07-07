# 数据集介绍

数据来源于 [Alibaba Cluster Trace Program v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)
，该数据集包含 4000 个机器上 8 天的运行数据，组织在 6 个文件中。

| Filename                                                                                                 | Size    | Comment                        |
|----------------------------------------------------------------------------------------------------------|---------|--------------------------------|
| [machine_meta.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_meta.tar.gz)       | 91 KB   | 包含机器的基本信息和事件的信息                |
| [machine_usage.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/machine_usage.tar.gz)     | 1.7 GB  | 包含机器的资源利用率信息                   |
| [container_meta.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/container_meta.tar.gz)   | 2.4 MB  | 包含容器的基本信息和事件的信息                |
| [container_usage.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/container_usage.tar.gz) | 27.2 GB | 包含每个容器的资源利用率信息                 |
| [batch_instance.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_instance.tar.gz)   | 19.7 GB | 包含 batch workloads 中实例的信息      |
| [batch_task.tar.gz](http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz)           | 142 MB  | 包含 batch workloads 中任务 DAG 的信息 |

在任务调度的相关研究中，只用到了 batch_task.tar.gz 部分。

> 注：阿里巴巴将数据分成了 online services（需要长时间运行的应用程序）和 batch workloads（批处理作业工作负载）两类，它们的主要区别在于服务时长。
> 针对这两类数据，阿里巴巴提供了 Sigma（用于调度在线服务），Fuxi（用于调度批处理作业）两种调度器。

## 数据集中的字段解释

该小节主要解释每个文件中表头的含义。值得注意的是，有一些共性的字段，单独介绍一下：

`time_stamp`, `start_time` 和 `end_time`：这三个字段的单位都是 “秒”，其值代表 实际时间 和 开始采样时间 的差值。开始时间是
0。

出于保密原因，文件对内存大小和磁盘大小等信息进行了缩放，使其值介于 0 到 100 之间，在该范围之外是无效值，如 -1 或 101。

### machine meta

```txt
+-------------------------------------------------------------------------------------+
| Field            | Type       | Label | Comment                                     |
+-------------------------------------------------------------------------------------+
| machine_id       | string     |       | uid of machine                              |
| time_stamp       | bigint     |       | time stamp, in second                       |
| failure_domain_1 | bigint     |       | one level of container failure domain       |
| failure_domain_2 | string     |       | another level of container failure domain   |
| cpu_num          | bigint     |       | number of cpu on a machine                  |
| mem_size         | bigint     |       | normalized memory size. [0, 100]            |
| status           | string     |       | status of a machine                         |
+-------------------------------------------------------------------------------------+
```

`failure_domain_1`：我们有多个不级别的故障域，在 v2018 只提供了其中的两个。对需要容错的应用程序，它们的实例应该分布在多个故障域中，这是一个枚举值。

### machine usage

```txt
+--------------------------------------------------------------------------------------------+
| Field            | Type       | Label | Comment                                            |
+--------------------------------------------------------------------------------------------+
| machine_id       | string     |       | uid of machine                                     |
| time_stamp       | double     |       | time stamp, in second                              |
| cpu_util_percent | bigint     |       | [0, 100]                                           |
| mem_util_percent | bigint     |       | [0, 100]                                           |
| mem_gps          | double     |       | normalized memory bandwidth, [0, 100]              |
| mkpi             | bigint     |       | cache miss per thousand instruction                |
| net_in           | double     |       | normarlized in coming network traffic, [0, 100]    |
| net_out          | double     |       | normarlized out going network traffic, [0, 100]    |
| disk_io_percent  | double     |       | [0, 100], abnormal values are of -1 or 101         |
+--------------------------------------------------------------------------------------------+
```

### container meta

```txt
+-----------------------------------------------------------------------------------------------------+
| Field           | Type       | Label | Comment                                                      |
+-----------------------------------------------------------------------------------------------------+
| container_id    | string     |       | uid of a container                                           |
| machine_id      | string     |       | uid of container's host machine                              |
| time_stamp      | bigint     |       |                                                              |
| app_du          | string     |       | containers with same app_du belong to same application group |
| status          | string     |       |                                                              |
| cpu_request     | bigint     |       | 100 is 1 core                                                |
| cpu_limit       | bigint     |       | 100 is 1 core                                                |
| mem_size        | double     |       | normarlized memory, [0, 100]                                 |
+-----------------------------------------------------------------------------------------------------+
```

`app_du`：处于同一个部署单元（deploy unit）中的容器共同对外提供一个服务，通常，它们应该分布在故障域中。

### container usage

```txt
+-----------------------------------------------------------------------------------------+
| Field            | Type       | Label | Comment                                         |
+-----------------------------------------------------------------------------------------+
| container_id     | string     |       | uid of a container                              |
| machine_id       | string     |       | uid of container's host machine                 |
| time_stamp       | double     |       | time stamp, in second                           |
| cpu_util_percent | bigint     |       |                                                 |
| mem_util_percent | bigint     |       |                                                 |
| cpi              | double     |       |                                                 |
| mem_gps          | double     |       | normalized memory bandwidth, [0, 100]           |
| mpki             | bigint     |       |                                                 |
| net_in           | double     |       | normarlized in coming network traffic, [0, 100] |
| net_out          | double     |       | normarlized out going network traffic, [0, 100] |
| disk_io_percent  | double     |       | [0, 100], abnormal values are of -1 or 101      |
+-----------------------------------------------------------------------------------------+
```

### batch task

```txt
+----------------------------------------------------------------------------------------+
| Field            | Type      | Label | Comment                                         |
+----------------------------------------------------------------------------------------+
| task_name       | string     |       | task name. unique within a job                  |
| instance_num    | bigint     |       | number of instances                             |
| job_name        | string     |       | job name                                        |
| task_type       | string     |       | task type                                       |
| status          | string     |       | task status                                     |
| start_time      | bigint     |       | start time of the task                          |
| end_time        | bigint     |       | end of time the task                            |
| plan_cpu        | double     |       | number of cpu needed by the task, 100 is 1 core |
| plan_mem        | double     |       | normalized memorty size, [0, 100]               |
+----------------------------------------------------------------------------------------+
```

* task_name 暗含了 DAG 信息，可以通过 task_name 逆向推导出 DAG
* task 之间有依赖关系，而 instance 没有提供相关依赖信息
* 一个 job 包含多个 task，每个 task 包含多个 instance
* 当且仅当 task 中的所有 instance 执行完毕后，才可以认为 task 执行完毕

如何根据 task_name 推导 DAG？

> 在 task_name 中，只有数字是我们需要注意的，第一个字母（`M`、`R`、`J`）是什么不重要。

- `task_Nzg3ODAwNDgzMTAwNTc2NTQ2Mw==`：独立任务，可单独执行
- `M1`：task1 是独立任务，不需要等待其他任务
- `M2_1`：task2 依赖 task1，即 task2 需要等待 task1 执行完成方可执行
- `R4_2`：task4 依赖 task2，即 task4 需要等待 task2 执行完成方可执行
- `M5_3_4`：task5 依赖 task3 和 task4

### batch instance

```txt
+-----------------------------------------------------------------------------------------------+
| Field           | Type       | Label | Comment                                                |
+-----------------------------------------------------------------------------------------------+
| instance_name   | string     |       | instance name of the instance                          |
| task_name       | string     |       | name of task to which the instance belong              |
| job_name        | string     |       | name of job to which the instance belong               |
| task_type       | string     |       | task type                                              |
| status          | string     |       | instance status                                        |
| start_time      | bigint     |       | start time of the instance                             |
| end_time        | bigint     |       | end time of the instance                               |
| machine_id      | string     |       | uid of host machine of the instance                    |
| seq_no          | bigint     |       | sequence number of this instance                       |
| total_seq_no    | bigint     |       | total sequence number of this instance                 |
| cpu_avg         | double     |       | average cpu used by the instance, 100 is 1 core        |
| cpu_max         | double     |       | max cpu used by the instance, 100 is 1 core            |
| mem_avg         | double     |       | average memory used by the instance (normalized)       |
| mem_max         | double     |       | max memory used by the instance (normalized, [0, 100]) |
+-----------------------------------------------------------------------------------------------+
```

* 同一个 job 中的 task_name 具有唯一性
* task_type 可细分为 12 种，但仅有部分 task_type 具有 DAG 信息
