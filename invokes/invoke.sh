#!/bin/bash

# 创建准备环境
kubectl get node -o wide | grep -v "NAME" | awk '{print $1,$6}' >hosts
cat hosts | while read n; do
    hostname=$(echo $n | awk '{print $1}')
    # ip=$(echo $n | awk '{print $2}')
    ssh -n $hostname "rm -rf ~/output && mkdir ~/output"
done

invokeTimes='3' # 调用次数
eventList=${1:-'cpu-cycles,L1-dcache-loads,branch-misses'}
interval=${2:-'500'} # ms
sleepTime=${4:-'1'}  # s
timestamp=$(date +%s.%N)

for funcName in $(faas-cli list | awk '{print $1}'); do
    # 跳过第一行
    if [ $funcName = 'Function' ]; then
        continue
    fi

    # 从第二行开始调用
    echo ''
    echo '============== '$funcName' =============='

    # 完成 invokeTimes 次函数调用
    for i in $(seq 1 $invokeTimes); do
        hostname=$(kubectl get pod -o wide -n openfaas-fn | grep $funcName | awk '{print substr($0, index($0, "k8s-"))}' | awk '{print $1}')
        url=$(kubectl describe pods -n openfaas-fn $funcName | grep 'containerd://' | awk '{print $3}')
        cid=${url#*//} # container id
        pid=$(ssh -n $hostname crictl inspect $cid | jq .info.pid)
        echo $funcName' on '$hostname' with pid number '$pid
        echo "hello" | faas-cli invoke $funcName # 开始调用函数，使用管道传递参数列表
        # 采集分支预测失误率数据
        ssh -n $hostname perf stat -a -e $eventList -I $interval -p $pid -o '~/output/'$hostname'_'$funcName'_'$timestamp'.out' sleep $sleepTime 2>&1 &
        # 采集响应时间数据
        invokeTime=$(date +%s.%N)
        reponseBody=$(curl -s $OPENFAAS_URL'/function/'$funcName'.openfaas-fn')
        endTime=$(date +%s.%N)
        startTime=$(echo $reponseBody | jq -r '.startTime') # 提取 JSON 中的字符串
        if [ ! $startTime ]; then
            startTime="''"
        fi
        echo "$funcName,$invokeTime,$startTime,$endTime" >>~/output/delay_and_execution_time_$hostname.out &
        # 采集 IO 数据
        ssh -n $hostname "blktrace -d /dev/sda -w $sleepTime -o - | blkparse -i - >~/output/io_$hostname.out &"
    done
done
