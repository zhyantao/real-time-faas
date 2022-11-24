import os
import random
import threading
import time
from multiprocessing import Process

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from Metrics import FunctionCount
from Metrics import Prometheus

import uuid

# 程序执行流程：entry -> runner -> run -> multi_process -> handler

reqest_col = ['time', 'req', 'platform']

result_col = ['actionName', 'invokeTime', 'startTime', 'endTime', 
              'schedule_latency', 'req', 'config', 'platform']

# 根据 action_name 的属性，启动若干线程
def handler(action_name, qps, config):
    uuidstring = config['uuidstring']
    cwd = config['cwd']
    threads = []
    print('starting  request')
    platform_name = config['platform_name']
    for i in range(qps):
        t = threading.Thread(target=start_client, args=(action_name, platform_name, uuidstring, cwd))
        threads.append(t)

    # 启动客户端
    start_time = time.time()
    config['first_req'] = start_time
    for i in range(qps):
        threads[i].start()

# 使用命令行工具执行 executor.sh，但是我这里不知道哪里有这个文件
def start_client(action_name, platform_name, uuidstring, cwd):
    command = "bash {cwd}/{platform_name}/executor.sh -a {action_name} -P {platform_name} -u {uuidstring}"
    command = command.format(cwd=cwd, action_name=action_name, platform_name=platform_name, uuidstring=uuidstring)
    os.system(command)

# 生成符合泊松分布的数据
def workload_generator(traffic=60, times=400):

    # 定义三组数据
    X = range(traffic)
    Y = []
    X1 = range(traffic, 2*traffic)
    Y1 = []
    X2 = range(2*traffic, 3*traffic)
    Y2 = []
    
    # 生成符合泊松分布的数据
    for k in X:
        p = stats.poisson.pmf(k, int(traffic/5)) * times
        Y.append(p)

    for k in X1:
        p = stats.poisson.pmf(k, int(traffic+traffic/4)) * times
        Y1.append(p)

    for k in X2:
        p = stats.poisson.pmf(k, int(2*traffic + traffic/2)) * times
        Y2.append(p)

    # 拼接三组泊松分布的数据
    x = np.concatenate((X, X1, X2))
    y = np.concatenate((Y, Y1, Y2))
    y = list(np.floor(y))

    # 将三组呈泊松分布的数据分割成对称的左右两部分（y_left_half 和 y_right_half）
    r1 = list(range(0, int(traffic/5)))
    r2 = list(range(traffic, int(traffic+traffic/4)))
    r3 = list(range(2*traffic, int(2*traffic + traffic/2)))

    reverse = np.concatenate((r1, r2, r3))
    y_left_half = y.copy()
    y_right_half = y.copy()
    for i in reverse:
        y_left_half[int(i)] = 0

    for i in range(traffic*3):
        if i not in reverse:
            y_right_half[i] = 0

    w=y[0:20]
    return x, w

# 针对 actions 使用多线程
def multi_process(actions, qps, config):
    request_threads = []
    if config['uuidstring'] == '':
        config['uuidstring']=uuid.uuid1()
    else:
        config['uuidstring'] = 'max-'+ str(uuid.uuid1())
    for action_name, params in actions.items():
        t = threading.Thread(target=handler, args=(action_name, qps, config))
        request_threads.append(t)

    random.shuffle(request_threads)
    total = len(request_threads)
    for i in range(total):
        request_threads[i].start()
    
    # for i in range(total):
    #     request_threads[i].join()


# 运行代码
def run(qps=5, mode='normal', platform_name='OpenFaas',last_state=False,uuids=''):
    start_time = time.time()
    cwd = os.getcwd()
    config = {"qps": qps, "first_req": '', "platform_name": platform_name, "last_state":last_state,"uuidstring":uuids,"cwd":cwd}

    # 读取 actions.yaml 文件的信息
    with open("./actions.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        lf_action = data_loaded.get("webservices")
        mf_action = data_loaded.get("MlI")
        bd_action = data_loaded.get("Big-Data")
        stream_action = data_loaded.get("Stream")
    try:
        p_web = Process(target=multi_process, args=(lf_action, qps, config))
        p_mf = Process(target=multi_process, args=(mf_action, qps, config))
        p_bigdata = Process(target=multi_process, args=(bd_action, qps, config))
        p_stream = Process(target=multi_process, args=(stream_action, qps, config))

        p_web.start()
        p_mf.start()
        p_bigdata.start()
        p_stream.start()

    except Exception:
        print('error...')
    end_time = time.time()
    record = 'start_time: ' + str(start_time) + \
         'end_time: ' + str(end_time) +'\n'

    with open('record'+platform_name+'.log', 'a+') as s:
        s.write(record)


# %%
def runner(namespace, platform_name, workload, period):
    last_state = False
    # manually control.
    threads_run = []
    # start function_instance counting
    fc = FunctionCount(namespace)
    print('start counting function instance')
    fc.get_pod_in_platform(platform_name, namespace)

    max_workload=int(max(workload)) 

    # start recording request.
    for i in range(len(workload)):
        qps = int(workload[i])
        if i == len(workload):
            last_state = True
        print('qps...', qps)
        time_now = time.time()
        df_req = pd.DataFrame({"time": [time_now], "req": [qps], "platform": [platform_name]})
        df_req.to_csv("request_"+platform_name+'.csv',header=False, index=False)
        if qps < 1:
            print('skiping')
            time.sleep(period)
            continue
        if qps ==max_workload:
            uuids='max'
        else:
            uuids=''
        t = threading.Thread(target=run, args=(qps, 'normal', platform_name, last_state, uuids))
        t.start()
        
        threads_run.append(t)
        time.sleep(period)
    for t in threads_run:
        if t.is_alive() :
            t.join()

    return
    
# 程序的入口程序
def entry():
    period = 5
    x, y = workload_generator(30,200) # 随机数据
    prom = Prometheus() # 运行时数据
    
    start = time.time() # 开始计时
    
    try:
        runner('openfaas-fn', 'OpenFaaS', y, period)
    except Exception:
        end = time.time()
    
    end = time.time() # 结束计时
    
    prom.run_prometheus_perf(start=start, end=end, platform='OpenFaaS', namespace='openfaas-fn')

    with open('runTime.log','w') as f:
        string = 'platform:'+ 'OpenFaaS', '+ start:'+ str(start), '+ end:'+ str(end)
        f.write(string)
        f.write("---")

# %%
os.chdir(os.path.dirname(__file__))
print(os.getcwd()) 
entry()